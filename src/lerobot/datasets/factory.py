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
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    # 某些视觉 backbone 在预训练时默认按 ImageNet 统计量做归一化。
    # 当配置要求 use_imagenet_stats 时，会把数据集元信息里的相机统计量覆盖成这里的固定值。
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """
    根据 policy 配置里的 delta_indices，生成 dataset 读取时要用的 delta_timestamps。

    这里的作用是把“以帧为单位的相对偏移”转换成“以秒为单位的相对偏移”。
    例如：
    - 如果数据集 fps=50
    - policy.observation_delta_indices = [-2, -1, 0]
    那么最终会得到：
    - [-0.04, -0.02, 0.0]

    后面 DatasetReader 会根据这些 delta_timestamps / delta_indices，
    在 __getitem__ 时把一条样本扩成一个时序窗口，而不只是读取当前时刻单帧。

    这个函数会遍历数据集里所有 feature key，然后根据 key 的类别决定应该使用哪组 delta：
    - reward 用 cfg.reward_delta_indices
    - action 用 cfg.action_delta_indices
    - observation.* 用 cfg.observation_delta_indices

    Args:
        cfg: policy 配置。里面定义了各模态要取哪些相对帧偏移。
        ds_meta: 数据集元信息，主要用它的 feature 列表和 fps。

    Returns:
        一个字典，key 是 feature 名称，value 是该特征对应的相对时间偏移列表。
        如果 policy 没有为任何特征配置 delta_indices，则返回 None。
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        # reward / action / observation 三类字段分别读取各自的 delta 配置。
        # 这里统一除以 fps，把“相对帧偏移”转换成“相对秒偏移”。
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    # 约定：如果没有任何时序窗口需求，就返回 None，而不是空 dict。
    # 这样 DatasetReader 可以直接把“是否启用时序查询”当成一个布尔开关判断。
    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """
    训练/评估前统一构造 dataset 对象。

    这个函数是 dataset 层的工厂入口，负责把“训练配置”转换成真正可读的数据集实例。
    它主要做四件事：

    1. 根据配置决定是否启用图像增强 / 图像变换。
    2. 先加载 dataset metadata，再结合 policy 配置解析 delta_timestamps。
    3. 根据是否开启 streaming，实例化 LeRobotDataset 或 StreamingLeRobotDataset。
    4. 如果配置要求使用 ImageNet 统计量，则覆盖相机模态的 mean/std。

    这里有个关键点：
    make_dataset() 不只是“new 一个 dataset”。
    它还负责把 policy 对时序窗口的需求提前翻译成 dataset 能理解的读取配置。
    也就是说，训练时 `__getitem__()` 到底只取当前帧，还是取前后多帧，
    在这里就已经决定了。

    Args:
        cfg: 训练主配置，里面同时包含 dataset 配置和 policy 配置。

    Raises:
        NotImplementedError: 当前代码路径下 MultiLeRobotDataset 还没有正式开放支持。

    Returns:
        构造好的单数据集对象；如果未来启用多数据集，这里也可能返回 MultiLeRobotDataset。
    """
    # 图像增强/变换是可选的。若关闭，就保持 None，后面 DatasetReader 不会额外处理图像。
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    # 目前最常见路径：repo_id 是单个字符串，表示读取一个数据集。
    if isinstance(cfg.dataset.repo_id, str):
        # 先只加载 metadata，不急着把所有 parquet / video 真的读进来。
        # 这是因为 resolve_delta_timestamps 只需要 features 和 fps。
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )

        # 根据 policy 的 delta_indices 和数据集 fps，得到真正给 reader 用的秒级偏移。
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        if not cfg.dataset.streaming:
            # 普通离线训练数据集：会在本地/Hub 上读取 parquet + 可选视频文件。
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            # 流式数据集：适合超大规模数据，不要求把整个 dataset 都以常规方式加载。
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        # repo_id 不是字符串时，表示用户可能传了多个数据集。
        # 但这一版代码里 MultiLeRobotDataset 仍然是停用状态，所以直接报错。
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        # 用固定的 ImageNet mean/std 覆盖相机统计量。
        # 这么做通常是为了匹配视觉 backbone 的预训练归一化习惯。
        # 注意这里只覆盖 camera_keys，对非图像特征没有影响。
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
