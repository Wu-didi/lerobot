#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CACHE_TENSORS_FILENAME = "dsrl_offline_cache.safetensors"
CACHE_METADATA_FILENAME = "dsrl_offline_cache.json"


def make_tensorboard_writer(log_dir: Path):
    """在 TensorBoard 依赖存在时创建 writer，否则安全降级为不写日志。

    TensorBoard 只是监控工具，不应该成为训练能否运行的硬依赖。因此这里采用
    lazy import：环境装了就写 event，没装就返回 None，让主训练继续执行。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        logging.warning("TensorBoard is not installed; skipping TensorBoard logging for %s", log_dir)
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def resolve_query_stride(query_stride: int | None, n_action_steps: int) -> int:
    """确定每个 episode 中离线 transition 的抽样间隔。

    DSRL actor 每次输出一个 action chunk，默认 stride 用 `n_action_steps`，
    表示“执行一个 chunk 后再做下一次 latent decision”。如果用户显式设置
    query_stride，就允许更密或更稀地抽样。
    """
    stride = n_action_steps if query_stride is None else query_stride
    if stride <= 0:
        raise ValueError("query_stride must be > 0")
    return stride


def _tensor_to_scalar(value: Any) -> Any:
    """把 dataset metadata/index 里的标量 tensor 转成普通 Python 值。

    LeRobot/HF dataset 返回的 index 有时是 tensor，有时是 Python 标量。统一转换后，
    后续构建 `abs_to_rel` 映射时不会因为类型不一致导致查不到样本。
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().reshape(-1).tolist()
    return value


def _value_to_bool(value: Any) -> bool:
    """把用户提供的 success 标签统一转换成 bool。

    success_feature 可能来自 episode metadata，也可能来自最后一帧 feature；
    它可能是 tensor、list 或普通 bool。这里统一成 bool，供 sparse reward 生成使用。
    """
    value = _tensor_to_scalar(value)
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Cannot convert an empty list to bool.")
        value = value[-1]
    return bool(value)


def _get_episode_row(episodes: Any, episode_index: int) -> Mapping[str, Any]:
    """从 pandas 或 HuggingFace 风格容器中读取一行 episode metadata。

    不同 LeRobot 版本里 `dataset.meta.episodes` 的类型不完全一致。这里兼容
    pandas `.iloc` 和 datasets 的 `episodes[index]`，避免真实数据集和测试数据集行为不一致。
    """
    if hasattr(episodes, "iloc"):
        return episodes.iloc[episode_index]
    return episodes[episode_index]


def _episode_has_column(episodes: Any, column_name: str) -> bool:
    """用兼容不同 LeRobot 版本的方式检查 episode metadata 是否包含某一列。"""
    if hasattr(episodes, "columns"):
        return column_name in episodes.columns
    if hasattr(episodes, "column_names"):
        return column_name in episodes.column_names
    return False


def build_episode_candidate_indices(
    dataset: LeRobotDataset, chunk_size: int, query_stride: int
) -> dict[int, list[int]]:
    """为每个 episode 构建 chunk-level 的 query-step 索引。

    Offline DSRL 不以每个 control frame 作为一个 transition，而是以 pi0.5 的
    action chunk 为粒度。这个函数只挑选那些从当前帧开始还能取满 `chunk_size`
    个动作的样本，避免训练时 action chunk 越过 episode 结尾。
    """

    # LeRobotDataset 会懒加载底层 HF dataset；这里显式激活，后面才能读取全局 index 列。
    reader = dataset.reader
    if reader.hf_dataset is None:
        reader.load_and_activate()

    # dataset metadata 使用 absolute frame index，而 `dataset[i]` 使用当前 reader 的 relative index。
    # 构建映射后，可以从 episode 的 absolute 范围安全找到可用于 Subset/DataLoader 的相对索引。
    abs_to_rel = {
        int(_tensor_to_scalar(abs_idx)): rel_idx for rel_idx, abs_idx in enumerate(reader.hf_dataset["index"])
    }
    # 如果用户没有指定 episodes，就遍历整个数据集；指定后只处理子集，便于小规模调试。
    if dataset.episodes is None:
        episode_ids = list(range(dataset.meta.total_episodes))
    else:
        episode_ids = list(dataset.episodes)

    candidates: dict[int, list[int]] = {}
    for ep_idx in episode_ids:
        # episode metadata 里记录了该 episode 在全局 frame 表中的起止位置。
        episode = _get_episode_row(dataset.meta.episodes, ep_idx)
        ep_from = int(episode["dataset_from_index"])
        ep_to = int(episode["dataset_to_index"])
        # 从 abs_idx 开始必须能取到完整 action chunk，所以最后一个合法起点是 ep_to - chunk_size。
        last_valid_abs = ep_to - chunk_size
        if last_valid_abs < ep_from:
            # 太短的 episode 不能形成一个完整 pi0.5 chunk，保留空列表方便后续统计。
            candidates[ep_idx] = []
            continue

        episode_indices = []
        # query_stride 决定 latent decision 的间隔；默认每执行 n_action_steps 后重新决策。
        for abs_idx in range(ep_from, last_valid_abs + 1, query_stride):
            rel_idx = abs_to_rel.get(abs_idx)
            if rel_idx is not None:
                episode_indices.append(rel_idx)
        candidates[ep_idx] = episode_indices

    return candidates


def resolve_episode_success_map(
    dataset: LeRobotDataset,
    candidate_indices_by_episode: Mapping[int, list[int]],
    success_feature: str | None = None,
    failed_episodes: set[int] | None = None,
    assume_all_success: bool = True,
) -> dict[int, bool]:
    """为每个 episode 解析一个成功/失败标签。

    V1 reward 采用 dsrl_pi0 风格 sparse reward，因此每个 episode 只需要一个
    success/failure 标签。真实 demo 数据通常全是成功轨迹，所以默认
    `assume_all_success=True`，同时保留 failed_episodes 和 success_feature 供后续混入失败数据。
    """

    failed_episodes = failed_episodes or set()
    success_by_episode: dict[int, bool] = {}

    reader = dataset.reader
    if reader.hf_dataset is None:
        reader.load_and_activate()

    abs_to_rel = {
        int(_tensor_to_scalar(abs_idx)): rel_idx for rel_idx, abs_idx in enumerate(reader.hf_dataset["index"])
    }

    for ep_idx, candidate_indices in candidate_indices_by_episode.items():
        # 手动指定失败 episode 的优先级最高，适合没有 success 字段但知道某些轨迹失败的情况。
        if ep_idx in failed_episodes:
            success_by_episode[ep_idx] = False
            continue

        # 优先从 episode metadata 读取 success，因为它是 episode-level 标签，语义最直接。
        if success_feature is not None and _episode_has_column(dataset.meta.episodes, success_feature):
            success_by_episode[ep_idx] = _value_to_bool(
                _get_episode_row(dataset.meta.episodes, ep_idx)[success_feature]
            )
            continue

        # 如果 success 是 frame-level feature，就取 episode 最后一帧作为整条轨迹的结果。
        if success_feature is not None and success_feature in dataset.meta.features:
            episode = _get_episode_row(dataset.meta.episodes, ep_idx)
            last_abs = int(episode["dataset_to_index"]) - 1
            rel_idx = abs_to_rel.get(last_abs)
            if rel_idx is not None:
                success_value = reader.hf_dataset[rel_idx][success_feature]
                success_by_episode[ep_idx] = _value_to_bool(success_value)
                continue

        # demo-only 数据通常都是成功示范；默认按成功处理才能产生 dsrl_pi0-style 最后一步 0 reward。
        if assume_all_success:
            success_by_episode[ep_idx] = True
            continue

        # 没有可训练 chunk 的 episode 视为失败/无效，不会进入 reward table 的训练样本。
        if len(candidate_indices) == 0:
            success_by_episode[ep_idx] = False
            continue

        # 如果用户关闭 assume_all_success，又没提供 success 信息，就显式报错，避免 reward 静默错误。
        raise ValueError(
            f"Could not resolve success for episode {ep_idx}. "
            f"Provide success_feature, failed_episodes, or set assume_all_success=True."
        )

    return success_by_episode


def build_sparse_reward_table(
    candidate_indices_by_episode: Mapping[int, list[int]],
    success_by_episode: Mapping[int, bool],
    discount: float,
) -> dict[str, torch.Tensor]:
    """创建 dsrl_pi0 风格 sparse reward 和折扣 return-to-go。

    对成功 episode：最后一个 query-step reward 为 0，之前都是 -1。
    对失败 episode：所有 query-step 都是 -1。
    这样和 dsrl_pi0 的“尽快成功，否则持续惩罚”形式对齐。
    """

    # 这些 list 会被转换成 tensor 并写入 cache，训练时用它们恢复 (s, z, r, s', done)。
    dataset_index: list[int] = []
    next_dataset_index: list[int] = []
    episode_index: list[int] = []
    reward: list[float] = []
    return_to_go: list[float] = []
    done: list[bool] = []

    for ep_idx, episode_indices in candidate_indices_by_episode.items():
        if len(episode_indices) == 0:
            continue

        # 默认每个 query-step 都有 -1 step cost，鼓励策略尽快达到成功终点。
        episode_rewards = [-1.0] * len(episode_indices)
        if success_by_episode.get(ep_idx, True):
            # 成功轨迹最后一步设为 0，表示达到目标后不再继续受惩罚。
            episode_rewards[-1] = 0.0

        episode_returns = [0.0] * len(episode_indices)
        running_return = 0.0
        # 反向累计 return-to-go，AWR fallback 和日志统计会用到。
        for idx in reversed(range(len(episode_indices))):
            running_return = episode_rewards[idx] + discount * running_return
            episode_returns[idx] = running_return

        for idx, rel_idx in enumerate(episode_indices):
            # next_dataset_index 指向下一个 latent decision；最后一步用 -1 表示 terminal。
            dataset_index.append(rel_idx)
            next_dataset_index.append(episode_indices[idx + 1] if idx + 1 < len(episode_indices) else -1)
            episode_index.append(ep_idx)
            reward.append(episode_rewards[idx])
            return_to_go.append(episode_returns[idx])
            done.append(idx == len(episode_indices) - 1)

    return {
        "dataset_index": torch.tensor(dataset_index, dtype=torch.int64),
        "next_dataset_index": torch.tensor(next_dataset_index, dtype=torch.int64),
        "episode_index": torch.tensor(episode_index, dtype=torch.int64),
        "reward": torch.tensor(reward, dtype=torch.float32),
        "return_to_go": torch.tensor(return_to_go, dtype=torch.float32),
        "done": torch.tensor(done, dtype=torch.bool),
    }


def build_awr_weights(
    return_to_go: torch.Tensor,
    beta: float,
    clip: float,
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
) -> torch.Tensor:
    """为 AWR fallback 构建基于 return 的行为克隆权重。

    IQL 是默认算法，但 AWR 可以作为更简单的 baseline。这里把 return 标准化后
    做指数加权，并裁剪到合理范围，避免单个高 return 样本让 actor loss 爆炸。
    """
    mean_t = torch.as_tensor(mean, dtype=return_to_go.dtype, device=return_to_go.device)
    std_t = torch.as_tensor(std, dtype=return_to_go.dtype, device=return_to_go.device).clamp_min(1e-6)
    normalized_return = (return_to_go - mean_t) / std_t
    return torch.exp(beta * normalized_return).clamp(min=1.0, max=clip)


class OfflineDSRLLatentDataset(Dataset):
    """把 LeRobotDataset 包装成带预计算 latent-noise 监督的离线 RL 数据集。

    这个 dataset 把原始 LeRobot 样本和 cache 里的 latent label/reward 绑定起来。
    训练脚本每次取到的是 `(raw_obs, next_raw_obs, labels)`，这样 IQL 可以计算
    Q(s, z)、reward 和 V(s')。
    """

    def __init__(self, dataset: LeRobotDataset, cache_tensors: Mapping[str, torch.Tensor]):
        """只保留 action-to-noise 反演成功的 cache 行。

        action-to-noise inversion 可能失败；`is_valid=False` 的样本不应该训练 critic/actor，
        否则 Q 会学习到错误的 latent action 标签。
        """
        self.dataset = dataset
        self.cache_tensors = cache_tensors
        self.valid_indices = torch.nonzero(cache_tensors["is_valid"], as_tuple=False).flatten().tolist()

    def __len__(self) -> int:
        """返回可用于训练的 latent transition 数量。"""
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
        """返回一个离线 RL transition。

        `index` 是 valid 样本中的位置；先映射回 cache row，再映射回 LeRobot dataset row。
        terminal transition 没有 next state 时复用当前 raw_item，训练时 done=1 会屏蔽 bootstrap。
        """
        cache_index = self.valid_indices[index]
        dataset_index = int(self.cache_tensors["dataset_index"][cache_index].item())
        raw_item = self.dataset[dataset_index]
        next_dataset_index = int(self.cache_tensors["next_dataset_index"][cache_index].item())
        next_raw_item = raw_item if next_dataset_index < 0 else self.dataset[next_dataset_index]
        labels = {
            key: tensor[cache_index]
            for key, tensor in self.cache_tensors.items()
            if key not in {"dataset_index", "next_dataset_index", "episode_index"}
        }
        labels["cache_index"] = torch.tensor(cache_index, dtype=torch.int64)
        labels["dataset_index"] = self.cache_tensors["dataset_index"][cache_index]
        labels["next_dataset_index"] = self.cache_tensors["next_dataset_index"][cache_index]
        labels["episode_index"] = self.cache_tensors["episode_index"][cache_index]
        return raw_item, next_raw_item, labels


def save_dsrl_offline_cache(
    output_dir: Path, cache_tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> None:
    """同时保存 cache tensor 和可读 metadata。

    tensor 数据用 safetensors，原因是它读写快、格式简单、不会执行任意代码；
    metadata 用 JSON，方便直接查看数据来源、样本数和 return 统计。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in cache_tensors.items()}
    save_file(tensors, str(output_dir / CACHE_TENSORS_FILENAME))
    with open(output_dir / CACHE_METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_dsrl_offline_cache(cache_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """加载已经预计算好的离线 DSRL cache。

    训练阶段只依赖这个 cache，不需要重新做昂贵的 action-to-noise inversion。
    这也是把流程拆成 precompute 和 train 两个阶段的主要原因。
    """
    cache_tensors = load_file(cache_dir / CACHE_TENSORS_FILENAME)
    with open(cache_dir / CACHE_METADATA_FILENAME) as f:
        metadata = json.load(f)
    return cache_tensors, metadata
