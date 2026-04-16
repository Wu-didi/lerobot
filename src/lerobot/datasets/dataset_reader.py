#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from collections.abc import Callable
from pathlib import Path

import datasets
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from lerobot.datasets.io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from lerobot.datasets.video_utils import decode_video_frames


class DatasetReader:
    """
    LeRobotDataset 的读侧实现。

    这个类专门负责“训练/评估时怎么按索引把一条样本读出来”。
    它管理三类核心状态：
    - ``hf_dataset``：底层 parquet 表格加载成的 Hugging Face Dataset
    - ``_absolute_to_relative_idx``：绝对索引到当前子集相对索引的映射
    - ``delta_indices``：时序窗口查询时使用的相对帧偏移

    LeRobotDataset 本身更像一个门面层，而真正的单样本读取逻辑都在这里。
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
    ):
        """
        初始化 reader 的读取配置。

        这里先保存 metadata、episode 过滤条件、视频后端、图像变换等配置，
        但不会立刻把底层 HF dataset 真的加载进来。
        真正的加载发生在 ``try_load()`` 或 ``load_and_activate()``。

        Args:
            meta: 数据集元信息对象。
            root: 本地数据集根目录。
            episodes: 可选，只读取指定 episode 子集；None 表示读取全部。
            tolerance_s: 时间同步容忍度。
            video_backend: 视频解码后端标识。
            delta_timestamps: 时序窗口配置，key 是 feature 名，value 是相对时间偏移列表。
            image_transforms: 应用在视觉模态上的图像变换。
        """
        self._meta = meta
        self.root = root
        self.episodes = episodes
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self._image_transforms = image_transforms

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None

        # 把秒级 delta_timestamps 转成帧级 delta_indices。
        # 这一步只依赖 fps 和配置，不需要真正读取 parquet。
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

    def try_load(self) -> bool:
        """
        尝试直接从本地缓存加载数据。

        如果 parquet 或视频文件不完整，会返回 False，让上层决定是否从 Hub 下载。
        """
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """从磁盘加载 HF dataset，并建立索引映射。"""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """
        建立“绝对索引 -> 当前子集相对索引”的映射表。

        当没有 episode 过滤时，HF dataset 的行号基本就能直接当索引用；
        但如果只选了一部分 episode，那么 parquet 里的全局 ``index`` 和
        当前子集里的相对位置就不再一致，这里需要做一次映射。
        """
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

    @property
    def num_frames(self) -> int:
        """当前选中 episode 子集中的总帧数。"""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """当前 reader 负责的 episode 数量。"""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """
        加载底层 parquet 数据表。

        这里得到的 hf_dataset 主要包含结构化字段：
        观测、动作、奖励、timestamp、索引等。
        视频帧本身不直接存这里，而是在后续 get_item() 里按 timestamp 去视频文件中查。
        """
        features = get_hf_features_from_features(self._meta.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """
        检查本地缓存是否足够支撑当前读取请求。

        这里不只检查 parquet 是否存在，还会检查：
        - 请求的 episode 是否都在本地
        - 如果数据集包含视频模态，对应视频文件是否也都存在
        """
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """
        返回当前所需 episode 对应的数据文件路径列表。

        这个结果主要给 ``snapshot_download`` 的 ``allow_patterns`` 用，
        让下载阶段尽量只拉当前确实需要的 parquet 和视频文件。
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """
        根据当前样本的绝对索引，计算时序窗口查询所需的其他索引。

        这里有两个关键行为：
        - 所有查询都被限制在当前 episode 内，绝不会跨 episode 取数据
        - 超出边界的位置不会报错，而是夹到边界帧，同时返回 ``*_is_pad`` 标记
        """
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """
        计算视频查询时要使用的时间戳。

        如果视频特征本身也配置了 delta 窗口，就用对应索引处的 timestamp；
        否则默认只查询当前样本时刻 ``current_ts``。
        """
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        从 HF dataset 中批量查询非视频特征。

        视频字段会被跳过，因为它们不在 parquet 里直接取，而是后面单独走视频解码逻辑。
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """
        按时间戳从视频文件中解码帧。

        注意这里查询的是“视频时间轴中的绝对时间”：
        - 数据表里的 ``timestamp`` 是 episode 内相对时间
        - 具体视频文件可能只覆盖某个 chunk/file 内的一段
        - 因此要先加上 ``from_timestamp`` 才能得到视频文件里的真实查询时间

        Note:
            当 DataLoader 使用多 worker 时，不要在主进程里同时混用另一套视频解码读取路径，
            否则这里可能触发底层解码器的段错误。
        """
        ep = self._meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, shifted_query_ts, self._tolerance_s, self._video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def get_item(self, idx) -> dict:
        """
        返回一条完整样本。

        这是 ``LeRobotDataset.__getitem__`` 真正委托的核心逻辑。
        处理顺序是：
        1. 先从 HF dataset 取出当前行
        2. 取出 episode_index 和全局绝对 index
        3. 如果配置了 delta 窗口，则额外查询前后时刻的非视频特征
        4. 如果有视频特征，再按 timestamp 去视频文件里解码帧
        5. 最后做图像变换，并把 task_index 映射成自然语言 task

        这里的 ``idx`` 是“当前已筛选子集中的相对索引”，不是 parquet 中的全局 ``index``。
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            # 时序窗口分两部分：
            # 1. query_indices：真正去取哪些帧
            # 2. padding：哪些位置原本越界，只是被边界帧替代了
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            # 图像增强放在样本读出之后统一做，这样无论图像来自 parquet 还是视频解码，
            # 上层都能拿到一致格式的视觉输入。
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        # task_index 是训练时更紧凑的整数表示；这里额外补回可读的任务文本，便于调试和分析。
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        # 如果数据集定义了更细粒度的 subtask，也同样映射回文本。
        if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name

        return item
