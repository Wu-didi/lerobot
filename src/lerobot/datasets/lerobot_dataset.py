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
import contextlib
import logging
from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.utils import (
    create_lerobot_dataset_card,
    get_safe_version,
    is_valid_version,
)
from lerobot.datasets.video_utils import (
    StreamingVideoEncoder,
    get_safe_default_codec,
    resolve_vcodec,
)
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

logger = logging.getLogger(__name__)


class LeRobotDataset(torch.utils.data.Dataset):
    """
    LeRobot 数据集的统一门面类。

    这个类把数据集系统拆成三块：
    - metadata：描述数据集结构、全局统计量、任务表、episode 边界等
    - reader：负责训练/评估时按索引随机读取样本
    - writer：负责录制/转换时顺序写入 frame 和 episode

    从使用方式看，它支持两种状态：
    - 只读模式：通过 ``LeRobotDataset(...)`` 打开一个已经存在的数据集
    - 写模式：通过 ``create()`` 或 ``resume()`` 得到一个可追加数据的对象

    外部代码通常只和 LeRobotDataset 交互，不直接操作 reader / writer。
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        """
        打开一个已经存在的数据集。

        这是训练和评估最常用的构造入口。它会先加载 metadata，再构造 reader，
        然后优先尝试使用本地缓存；如果本地数据不完整，再从 Hugging Face Hub 下载。

        这个构造函数默认服务于“读取已有数据集”的场景。
        如果你要新建数据集或者在已有数据集末尾继续录制，推荐使用
        :meth:`create` 或 :meth:`resume`。

        这个类底层管理的主要文件类型有三种：
        - ``meta/``：info、stats、tasks、episodes 等元信息
        - ``data/``：parquet 表格，存状态、动作、时间戳、索引等结构化数据
        - ``videos/``：视频文件，读取时按 timestamp 解码出对应帧

        Args:
            repo_id: 数据集仓库 ID，例如 ``user/my_dataset``。
            root: 本地路径。提供时优先从这里读取或下载到这里；不提供时使用默认缓存。
            episodes: 若不为 None，只加载指定的 episode 子集。
            image_transforms: 可选图像变换，应用在视觉模态上。
            delta_timestamps: 可选时序窗口配置，告诉 reader 读取当前样本前后哪些时间偏移。
            tolerance_s: 时间同步容忍度，用于校验 timestamp/fps，也用于视频解码时容忍误差。
            revision: Hub 版本号，可以是 branch/tag/commit。
            force_cache_sync: 为 True 时，即便本地有缓存也优先同步。
            download_videos: 下载数据集时是否把视频一并下载。
            video_backend: 视频解码后端。
            batch_encoding_size: 写模式下的旧参数，仅为了兼容老调用方式。
            vcodec: 写模式下的视频编码器配置，仅为了兼容老调用方式。
            streaming_encoding: 写模式下是否启用实时编码，仅为了兼容老调用方式。
            encoder_queue_maxsize: streaming 编码时每个相机的队列上限。
            encoder_threads: 视频编码线程数。

        Note:
            把写模式参数直接传给 ``__init__`` 已经过时。
            新建数据集请用 :meth:`create`，继续录制请用 :meth:`resume`。
        """
        super().__init__()
        self.repo_id = repo_id  # 'zsx/fold_clothes0402'
        self._requested_root = Path(root) if root else None
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self._video_backend = video_backend if video_backend else get_safe_default_codec()
        self._batch_encoding_size = batch_encoding_size
        self._vcodec = resolve_vcodec(vcodec)
        self._encoder_threads = encoder_threads

        if self._requested_root is not None:
            self._requested_root.mkdir(exist_ok=True, parents=True)

        # 先加载 metadata。metadata 会决定真正的数据集根目录、特征、统计量、episode 索引等。
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self._requested_root, self.revision, force_cache_sync=force_cache_sync
        )
        self.root = self.meta.root
        self.revision = self.meta.revision

        # 先构造 reader，但 HF dataset 本体是否立刻激活，取决于本地缓存是否完整。
        self.reader = DatasetReader(
            meta=self.meta,
            root=self.root,
            episodes=episodes,
            tolerance_s=tolerance_s,
            video_backend=self._video_backend,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
        )

        # 优先尝试使用本地已有数据；如果本地不完整，再触发下载。
        if force_cache_sync or not self.reader.try_load():
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            self._download(download_videos)
            self.reader.load_and_activate()

        # 这段是兼容旧接口的“半写模式”路径。
        # 新代码不建议再依赖这里，应该显式调用 create()/resume()。
        _has_write_params = streaming_encoding or batch_encoding_size != 1
        if _has_write_params:
            import warnings

            warnings.warn(
                "Passing write-mode parameters (streaming_encoding, batch_encoding_size) to "
                "LeRobotDataset.__init__() is deprecated. Use LeRobotDataset.resume() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            streaming_enc = None
            if streaming_encoding and len(self.meta.video_keys) > 0:
                streaming_enc = self._build_streaming_encoder(
                    self.meta.fps, self._vcodec, encoder_queue_maxsize, encoder_threads
                )
            self.writer = DatasetWriter(
                meta=self.meta,
                root=self.root,
                vcodec=self._vcodec,
                encoder_threads=encoder_threads,
                batch_encoding_size=batch_encoding_size,
                streaming_encoder=streaming_enc,
                initial_frames=self.meta.total_frames,
            )
        else:
            self.writer = None

        # 写模式只有 finalize() 之后才允许安全读取。
        self._is_finalized = False

    # ── Writer guard ──────────────────────────────────────────────────

    def _require_writer(self, method_name: str) -> None:
        """确保当前 dataset 处于可写状态。"""
        if self.writer is None:
            raise RuntimeError(
                f"Cannot call '{method_name}()' on a read-only dataset. "
                f"Use LeRobotDataset.create() for new recording or "
                f"LeRobotDataset.resume() for resume recording."
            )

    # ── Reader guard ──────────────────────────────────────────────────

    def _ensure_reader(self) -> DatasetReader:
        """
        按需构造 reader。

        create()/resume() 生成的对象一开始通常只负责写，不需要 reader。
        只有真正进入读取路径时，才在这里延迟初始化。
        """
        if self.reader is None:
            self.reader = DatasetReader(
                meta=self.meta,
                root=self.root,
                episodes=self.episodes,
                tolerance_s=self.tolerance_s,
                video_backend=self._video_backend,
                delta_timestamps=self.delta_timestamps,
                image_transforms=self.image_transforms,
            )
        return self.reader

    @staticmethod
    def _build_streaming_encoder(
        fps: int,
        vcodec: str,
        encoder_queue_maxsize: int,
        encoder_threads: int | None,
    ) -> StreamingVideoEncoder:
        """构造实时视频编码器。"""
        return StreamingVideoEncoder(
            fps=fps,
            vcodec=vcodec,
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=None,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )

    # ── Metadata properties ───────────────────────────────────────────

    @property
    def fps(self) -> int:
        """数据采集时使用的 fps。"""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """
        当前选中 episode 子集中的总帧数。

        在写模式下，这里不强行初始化 reader，而是直接依赖 metadata 里的累计值。
        """
        if self.reader is None:
            return self.meta.total_frames
        return self.reader.num_frames

    @property
    def num_episodes(self) -> int:
        """
        当前选中的 episode 数量。

        这里同样兼容“只写不读”的 dataset 状态。
        """
        if self.reader is None:
            return self.meta.total_episodes
        return self.reader.num_episodes

    @property
    def features(self) -> dict[str, dict]:
        """返回 feature 定义表，描述每个字段的类型、shape、语义。"""
        return self.meta.features

    @property
    def hf_dataset(self) -> datasets.Dataset:
        """
        返回底层 Hugging Face Dataset 对象。

        这是 parquet 数据表的直接入口。若 reader 尚未激活，这里会触发一次加载。
        """
        self.reader = self._ensure_reader()
        if self.reader.hf_dataset is None:
            self.reader.load_and_activate()
        return self.reader.hf_dataset

    # ── Writer-delegated methods ──────────────────────────────────────

    def add_frame(self, frame: dict) -> None:
        """
        向当前 episode 缓冲区追加一帧。

        这是录制路径最细粒度的写接口。真正的落盘不会在这里完成，
        而是先进入 writer 的 episode_buffer，等到 save_episode() 再统一写 parquet 和视频。

        Args:
            frame: 一帧数据，必须包含 feature 对应字段以及 ``task``。

        Raises:
            RuntimeError: 当前是只读 dataset 时抛出。
        """
        self._require_writer("add_frame")
        self.writer.add_frame(frame)

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        """
        把当前 episode 缓冲区正式保存到磁盘。

        这个调用会触发写主数据表、编码视频、更新 episode metadata 和 stats。
        如果没有显式传入 ``episode_data``，就默认使用 add_frame() 一路积累的内部缓冲区。

        Args:
            episode_data: 可选，手工构造好的 episode 字典。
            parallel_encoding: 多相机时是否并行编码视频。

        Raises:
            RuntimeError: 当前是只读 dataset 时抛出。
        """
        self._require_writer("save_episode")
        self.writer.save_episode(episode_data, parallel_encoding)

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """
        丢弃当前还没保存的 episode 缓冲区。

        常见于录制失败、人工重录、或者中途终止时，放弃当前 episode 而不落盘。

        Args:
            delete_images: 是否连同当前 episode 的临时图像文件一起删掉。

        Raises:
            RuntimeError: 当前是只读 dataset 时抛出。
        """
        self._require_writer("clear_episode_buffer")
        self.writer.clear_episode_buffer(delete_images)

    def has_pending_frames(self) -> bool:
        """判断当前是否还有尚未保存的帧。"""
        if self.writer is None:
            return False
        return self.writer.episode_buffer is not None and self.writer.episode_buffer["size"] > 0

    def finalize(self):
        """
        完成写模式数据集的最终收尾。

        finalize() 的职责是：
        - 把 writer 里尚未刷盘的内容全部落盘
        - 关闭 parquet writer / 编码器等资源
        - 把 dataset 标记为“可以安全读取”

        如果录制完不调用 finalize()，数据文件尾部元信息可能不完整，
        后续读取就可能失败或得到无效数据。

        这个方法是幂等的，多次调用不会重复破坏状态。
        """
        if self._is_finalized:
            return
        if self.writer is not None:
            self.writer.finalize()
        self._is_finalized = True

    # ── Core Dataset methods ──────────────────────────────────────────

    def __len__(self):
        """返回当前 dataset 可见的样本数。"""
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        """
        按索引返回一条训练/评估样本。

        这是 PyTorch DataLoader 真正会调用的接口。
        复杂逻辑都不在这里手写，而是交给 DatasetReader.get_item()：
        - 从 parquet 读取基础字段
        - 根据 delta_timestamps 展开时间窗口
        - 解码对应视频帧
        - 应用图像变换

        注意：如果当前 dataset 仍处于录制中且还没 finalize()，
        这里会拒绝读取，避免训练侧看到半成品数据。

        Args:
            idx: dataset 内部索引。若指定了 episodes 子集，它是过滤后的相对索引。

        Returns:
            一个字典，key 是 feature 名，value 是该样本对应的张量/值。

        Raises:
            RuntimeError: 写模式尚未 finalize 时抛出。
        """
        if self.writer is not None and not self._is_finalized:
            raise RuntimeError(
                "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
            )
        reader = self._ensure_reader()
        if reader.hf_dataset is None:
            # 这是写模式 finalize 之后第一次读取时的“一次性激活”路径。
            reader.load_and_activate()
        return reader.get_item(idx)

    def select_columns(self, column_names: str | list[str]):
        """
        只选择底层 HF dataset 的部分列。

        适合做轻量分析，例如只取动作列，而不触发完整样本读取流程。
        """
        return self.hf_dataset.select_columns(column_names)

    def get_raw_item(self, idx) -> dict:
        """
        直接返回原始表格行，不做高级读取逻辑。

        和 ``__getitem__`` 不同，这里不会：
        - 展开 delta_timestamps
        - 解码视频帧
        - 应用图像变换

        这个接口更适合调试 dataset 底层存储内容。
        """
        return self.hf_dataset[idx]

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            f"}})"
        )

    # ── Hub methods (stay on facade) ──────────────────────────────────

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        """
        把当前数据集上传到 Hugging Face Hub。

        这个接口会负责：
        - 必要时创建 dataset repo
        - 上传本地文件夹
        - 生成并推送 dataset card
        - 按当前代码库版本打 tag

        Args:
            branch: 可选目标分支；不存在时会先创建。
            tags: dataset card 上附加的标签。
            license: dataset card 使用的许可证标识。
            tag_version: 是否为当前代码版本创建 Git tag。
            push_videos: 为 False 时跳过 ``videos/`` 目录。
            private: 是否创建私有 dataset repo。
            allow_patterns: 限制上传文件的 glob 规则。
            upload_large_folder: 超大数据集时改用 ``upload_large_folder``。
            **card_kwargs: 透传给 dataset card 生成逻辑的其他参数。
        """
        # images/ 往往只是中间产物，不需要上传；videos/ 是否上传由参数控制。
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, repo_id=self.repo_id, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def _download(self, download_videos: bool = True) -> None:
        """
        从 Hub 下载数据集到本地。

        如果当前对象只请求了部分 episode，这里会尽量只下载这些 episode 对应的数据和视频文件，
        避免把整套大数据全部拉到本地。
        """
        ignore_patterns = None if download_videos else "videos/"
        files = None
        if self.episodes is not None:
            # reader 在 __init__ 里已经创建好，这里复用它来计算“只下载哪些文件”。
            files = self.reader.get_episodes_file_paths()

        if self._requested_root is None:
            # 没指定 root 时，下载到 revision-safe 的共享 snapshot cache。
            self.meta.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=files,
                    ignore_patterns=ignore_patterns,
                )
            )
        else:
            # 指定了 root 时，直接把数据实体化到用户提供的目录。
            self._requested_root.mkdir(exist_ok=True, parents=True)
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self._requested_root,
                allow_patterns=files,
                ignore_patterns=ignore_patterns,
            )
            self.meta.root = self._requested_root

        # metadata.root 是最终单一可信来源，dataset 和 reader 都同步到这里。
        self.root = self.meta.root
        self.reader.root = self.meta.root

    # ── Class constructors ────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        metadata_buffer_size: int = 10,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ) -> "LeRobotDataset":
        """
        从零创建一个全新的可写数据集。

        这是录制数据时的正式入口。它不会走 ``__init__`` 的读取路径，
        而是直接：
        1. 创建 metadata 骨架和 ``meta/info.json``
        2. 初始化 writer
        3. 返回一个“只写模式”的 LeRobotDataset

        标准使用顺序通常是：
        - ``add_frame()`` 累积一帧帧数据
        - ``save_episode()`` 保存一个 episode
        - ``finalize()`` 完成收尾

        Args:
            repo_id: 数据集仓库 ID。
            fps: 采集频率。
            features: feature 定义表。
            root: 本地保存路径。
            robot_type: 可选，写入 metadata 的机器人类型。
            use_videos: 是否把视觉模态保存为视频。
            tolerance_s: 时间同步容忍度。
            image_writer_processes: 异步写图像的子进程数。
            image_writer_threads: 异步写图像的线程数。
            video_backend: 以后读视频时默认使用的后端。
            batch_encoding_size: 多少个 episode 聚合后再批量编码视频。
            vcodec: 视频编码器。
            metadata_buffer_size: metadata parquet 的缓冲区大小。
            streaming_encoding: 是否边采边编码视频。
            encoder_queue_maxsize: streaming 编码时每相机队列大小。
            encoder_threads: 编码线程数。

        Returns:
            一个处于写模式的 LeRobotDataset。
        """
        vcodec = resolve_vcodec(vcodec)
        obj = cls.__new__(cls)
        # create() 直接走 metadata.create()，不会触发“读取现有数据集”的初始化逻辑。
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
            metadata_buffer_size=metadata_buffer_size,
        )
        obj.repo_id = obj.meta.repo_id
        obj._requested_root = obj.meta.root
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj._batch_encoding_size = batch_encoding_size
        obj._vcodec = vcodec
        obj._encoder_threads = encoder_threads

        # 新建数据集时一开始只负责写，不需要 reader。
        obj.reader = None

        # 真正把“可写能力”接上的是 DatasetWriter。
        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(fps, vcodec, encoder_queue_maxsize, encoder_threads)
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            vcodec=vcodec,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
        )

        # 如果配置了异步图片写入，这里直接启动后台 writer。
        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj

    @classmethod
    def resume(
        cls,
        repo_id: str,
        root: str | Path | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ) -> "LeRobotDataset":
        """
        在一个已有数据集后面继续追加录制。

        这个入口适用于“数据集已经存在，本次只是在末尾接着加 episode”。
        它会读取已有 metadata，保留之前的 total_frames / total_episodes，
        然后创建一个新的 DatasetWriter，从现有末尾继续写。

        Args:
            repo_id: 已存在的数据集仓库 ID。
            root: 本地数据集目录。这里必须显式提供。
            tolerance_s: 时间同步容忍度。
            revision: 版本号。
            force_cache_sync: 是否强制先和 Hub 同步 metadata。
            video_backend: 视频解码后端。
            batch_encoding_size: 视频批量编码粒度。
            vcodec: 视频编码器。
            image_writer_processes: 异步写图像的子进程数。
            image_writer_threads: 异步写图像的线程数。
            streaming_encoding: 是否启用实时视频编码。
            encoder_queue_maxsize: streaming 编码队列大小。
            encoder_threads: 编码线程数。

        Returns:
            一个处于写模式、并且会在原数据集末尾继续写入的 LeRobotDataset。
        """
        if not root:
            raise ValueError(
                "resume() requires an explicit 'root' directory because it creates a DatasetWriter. "
                "Writing into the revision-safe Hub snapshot cache (used when root=None) would corrupt "
                "the shared cache. Please provide a local directory path."
            )
        vcodec = resolve_vcodec(vcodec)
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root)
        obj.revision = revision if revision else CODEBASE_VERSION
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend else get_safe_default_codec()
        obj._batch_encoding_size = batch_encoding_size
        obj._vcodec = vcodec
        obj._encoder_threads = encoder_threads

        if obj._requested_root is not None:
            obj._requested_root.mkdir(exist_ok=True, parents=True)

        # resume 的关键不是读完整数据，而是把已有 metadata 状态先接回来，
        # 这样 writer 才知道从第几个 episode、哪个全局 frame 开始续写。
        obj.meta = LeRobotDatasetMetadata(
            obj.repo_id, obj._requested_root, obj.revision, force_cache_sync=force_cache_sync
        )
        obj.root = obj.meta.root

        # 继续录制时默认也不需要 reader，仍然采用按需初始化策略。
        obj.reader = None

        # initial_frames 很关键：它保证续写数据时 index / 时间范围能接上原数据集。
        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                obj.meta.fps, vcodec, encoder_queue_maxsize, encoder_threads
            )
        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            vcodec=vcodec,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
            initial_frames=obj.meta.total_frames,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj
