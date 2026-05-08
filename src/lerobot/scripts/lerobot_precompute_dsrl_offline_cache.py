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

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dsrl_pi05.configuration_dsrl_pi05 import DSRLPi05Config
from lerobot.policies.dsrl_pi05.offline_utils import (
    build_episode_candidate_indices,
    build_sparse_reward_table,
    make_tensorboard_writer,
    resolve_episode_success_map,
    resolve_query_stride,
    save_dsrl_offline_cache,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.logging_utils import AverageMeter
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

# import debugpy
# debugpy.listen(12345)
# print("wait debug")
# debugpy.wait_for_client()
# print("Debugger attached")

@dataclass
class DSRLOfflineDatasetConfig:
    """离线 DSRL cache 生成阶段的数据集配置。"""

    # LeRobot 数据集 repo id，用于读取 metadata 和 feature 定义。
    repo_id: str
    # 本地数据集根目录；为空时走 LeRobot/HF 默认缓存路径。
    root: Path | None = None
    # 数据集 revision，保持和 LeRobotDataset 接口一致。
    revision: str | None = None
    # 只处理指定 episode；调试时可以传小子集，正式训练时保持 None 跑全量。
    episodes: list[int] | None = None
    # 反演很耗显存，默认 batch_size 较小，用户可根据 GPU 调整。
    batch_size: int = 4
    # 真实机器人数据通常 IO 不是瓶颈，默认 0 便于排查 worker 问题。
    num_workers: int = 0
    # 可选 success 标签字段；第一版没有标签时默认把 demo 视为成功。
    success_feature: str | None = None
    # 采集的折衣服 demo 通常是成功示范，因此默认生成成功轨迹 sparse reward。
    assume_all_success: bool = True
    # 手动指定失败 episode，可在没有 success_feature 时修正 reward。
    failed_episodes: list[int] = field(default_factory=list)


@dataclass
class DSRLOfflineCacheConfig:
    """action-to-noise 反演和 cache 写盘的完整脚本配置。"""

    # policy 配置里包含 base pi0.5 路径、反演步数、reward discount 等核心超参。
    policy: DSRLPi05Config
    # 数据集配置独立出来，CLI 参数会呈现为 --dataset.xxx。
    dataset: DSRLOfflineDatasetConfig
    # cache 输出目录，后续训练脚本通过 --cache_dir 读取这里。
    output_dir: Path = Path("outputs/dsrl/offline_cache")
    # 固定随机种子，保证反演随机初始化和 dataloader 行为可复现。
    seed: int = 0
    # 调试用：限制最多处理多少个 batch；正式训练保持 None。
    max_batches: int | None = None
    # 是否写 TensorBoard precompute 指标。
    tensorboard: bool = True
    # 可选 TensorBoard 目录；为空时写到 output_dir/runs/precompute。
    tensorboard_log_dir: Path | None = None


def make_lerobot_dataset(cfg: DSRLOfflineCacheConfig) -> LeRobotDataset:
    """根据 policy 的 delta_timestamps 构建 LeRobotDataset。

    DSRL 需要和 pi0.5 base policy 完全一致的观测窗口/action chunk；
    因此先从 dataset metadata 和 policy config 解析 delta_timestamps。
    """
    # metadata 读取轻量信息，不会立刻加载全部样本。
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    # delta_timestamps 决定每个 index 取哪些历史观测和未来 action。
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    return LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
    )


def make_dsrl_preprocessor(policy, dataset_stats):
    """构建和 base pi0.5 一致的 processor。

    这里传入 base checkpoint 路径，是为了复用训练 base policy 时保存的 processor 配置；
    测试 double 可能没有这个参数，所以下面保留兼容分支。
    """
    pretrained_path = str(policy.config.base_policy_path) if policy.config.base_policy_path is not None else None
    try:
        return make_pre_post_processors(
            policy.config,
            pretrained_path=pretrained_path,
            dataset_stats=dataset_stats,
        )
    except TypeError:
        # 测试替身可能不接受 pretrained_path 参数，退回最小调用形式。
        return make_pre_post_processors(policy.config, dataset_stats=dataset_stats)


@parser.wrap()
def main(cfg: DSRLOfflineCacheConfig) -> None:
    """生成离线 DSRL 训练所需的 latent cache。

    主要流程是：构建 query-step transition、生成 sparse reward、把 demo action
    反演成 latent noise、过滤重建失败样本，最后写成 safetensors + JSON metadata。
    """
    init_logging()
    set_seed(cfg.seed)

    # 先构建数据集，后续 policy factory 需要 dataset.meta 来补全 feature 信息。
    dataset = make_lerobot_dataset(cfg)
    # 离线 DSRL 训练的 action 不是单步机器人动作，而是 pi0.5 flow 的 latent noise chunk。
    # pi0.5 每次根据一个 observation 生成一段 action chunk，机器人会连续执行其中的 n_action_steps 步；
    # 因此 RL 里的一个 transition 应该对应“一次 latent decision”，而不是数据集里的每一帧。
    # 如果每一帧都反演 noise，会产生大量高度重叠的 action chunk，训练样本看似变多，
    # 但实际信息重复、cache 生成更慢，reward/done 也更难和 chunk 执行边界对齐。
    # query_stride 就是 latent decision 的抽样间隔：默认等于 n_action_steps，
    # 表示执行完上一段 chunk 后，再在新的 frame 上重新选择下一段 latent noise。
    query_stride = resolve_query_stride(cfg.policy.query_stride, cfg.policy.n_action_steps)
    # candidate_indices_by_episode 保存每条 episode 中可以作为 latent decision 起点的 dataset index。
    # 后续会用这些 index 构造 (s, z, r, s', done)：当前 index 是 s，
    # 同一 episode 里的下一个 candidate index 是 s'，最后一个 candidate index 对应 terminal。
    # 同时还要保证从该 index 开始能取到完整 chunk_size 的未来动作，否则无法反演完整 noise label。
    candidate_indices_by_episode = build_episode_candidate_indices(
        dataset=dataset,
        chunk_size=cfg.policy.chunk_size,
        query_stride=query_stride,
    )
    # 解析每条轨迹是否成功；第一版 sparse reward 只需要 episode-level 成败。
    success_by_episode = resolve_episode_success_map(
        dataset=dataset,
        candidate_indices_by_episode=candidate_indices_by_episode,
        success_feature=cfg.dataset.success_feature,
        failed_episodes=set(cfg.dataset.failed_episodes),
        assume_all_success=cfg.dataset.assume_all_success,
    )
    # success_by_episode 是一个 dict，key 是 episode index，value 是 bool 成败标签。
    # 例如：{0: True, 1: False, ...}。
    # 第一版折衣服数据集没有 success_feature，所以默认所有 episode 都是成功示范；
    # 如果后续混入失败数据，可以通过 failed_episodes 明确指定哪些 episode 失败。
    # reward_table 存储 dataset_index、next_dataset_index、reward、done、return_to_go。
    reward_table = build_sparse_reward_table(
        candidate_indices_by_episode=candidate_indices_by_episode,
        success_by_episode=success_by_episode,
        discount=cfg.policy.discount,
    )

    if reward_table["dataset_index"].numel() == 0:
        raise ValueError("No valid query-step samples were found for DSRL offline cache generation.")

    # cache 目录提前创建，这样 TensorBoard 和最终 safetensors 都有确定落点。
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = None
    if cfg.tensorboard:
        tb_writer = make_tensorboard_writer(cfg.tensorboard_log_dir or cfg.output_dir / "runs" / "precompute")

    # make_policy 会实例化 DSRLPi05Policy，并加载冻结 base pi0.5。
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    # processor 负责图像、语言、state/action 归一化，必须与 pi0.5 训练时一致。
    preprocessor, _ = make_dsrl_preprocessor(policy, dataset.meta.stats)

    # 只遍历 reward_table 里合法的 query-step 样本，而不是遍历全部 frame。
    candidate_indices = reward_table["dataset_index"].tolist()
    dataloader = DataLoader(
        Subset(dataset, candidate_indices),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )

    # reward_lists 保存已经对齐好的 RL transition 字段。
    reward_lists = {key: [] for key in reward_table}
    # tensor_lists 保存反演得到的 latent label 和质量信息。
    tensor_lists = {
        "noise_label": [],
        "recon_error": [],
        "absolute_index": [],
        "task_index": [],
    }

    recon_meter = AverageMeter("recon_error")
    # offset 表示已经处理了 reward_table 中多少个 query-step，用于切片同步。
    offset = 0
    progress = tqdm(dataloader, desc="Precomputing DSRL cache")
    for batch_idx, raw_batch in enumerate(progress):
        if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
            break

        batch_size = raw_batch["index"].shape[0]
        # raw batch 先经过 pi0.5 processor，得到模型可直接消费的 tensor。
        processed_batch = preprocessor(raw_batch)
        target_actions = processed_batch[ACTION]
        # 核心步骤：固定 pi0.5 decoder，优化 latent noise 来重建 demo action。
        noise_label, recon_error = policy.invert_actions_to_noise(processed_batch, target_actions)

        for key, value in reward_table.items():
            # reward_table 顺序和 Subset(dataset, candidate_indices) 顺序一致，所以用 offset 对齐。
            reward_lists[key].append(value[offset : offset + batch_size].detach().cpu())

        # noise_label 展平成 `[B, latent_action_dim]`，方便训练时直接送入 MLP critic。
        tensor_lists["noise_label"].append(noise_label.reshape(batch_size, -1).detach().cpu())
        tensor_lists["recon_error"].append(recon_error.detach().cpu())
        tensor_lists["absolute_index"].append(raw_batch["index"].detach().cpu().to(dtype=torch.int64))
        tensor_lists["task_index"].append(raw_batch["task_index"].detach().cpu().to(dtype=torch.int64))

        offset += batch_size
        recon_meter.update(recon_error.mean().item(), n=batch_size)
        if tb_writer is not None:
            # 这些指标用于判断反演质量；valid_ratio 要等全量 recon_error 统计完并确定阈值后再计算。
            tb_writer.add_scalar("precompute/recon_error", recon_error.mean().item(), offset)
            tb_writer.add_scalar("precompute/recon_error_avg", recon_meter.avg, offset)
        progress.set_postfix(
            recon_error=f"{recon_meter.avg:.5f}",
            total=offset,
        )

    # 把分 batch 收集的 list 合并成最终 cache tensor。
    cache_tensors = {
        key: torch.cat(values, dim=0) if len(values) > 0 else reward_table[key][:0]
        for key, values in reward_lists.items()
    }
    for key, values in tensor_lists.items():
        if len(values) == 0:
            raise ValueError("Offline cache generation did not produce any samples.")
        cache_tensors[key] = torch.cat(values, dim=0)

    # 先完成所有样本反演，再基于全量 recon_error 分布统一求阈值。
    # 这样 threshold 来自当前数据集自己的反演质量，而不是预先拍一个固定值。
    recon_errors = cache_tensors["recon_error"].to(dtype=torch.float32)
    if cfg.policy.latent_recon_threshold is None:
        quantile_threshold = torch.quantile(recon_errors, cfg.policy.latent_recon_quantile)
        resolved_threshold = quantile_threshold
        threshold_source = "auto_quantile"
        if cfg.policy.latent_recon_max_threshold is not None:
            # 可选绝对上限用于防止整体反演都很差时，分位数阈值被异常抬得过高。
            max_threshold = torch.tensor(
                cfg.policy.latent_recon_max_threshold,
                dtype=resolved_threshold.dtype,
                device=resolved_threshold.device,
            )
            resolved_threshold = torch.minimum(resolved_threshold, max_threshold)
            threshold_source = "auto_quantile_with_max"
    else:
        quantile_threshold = torch.quantile(recon_errors, cfg.policy.latent_recon_quantile)
        resolved_threshold = torch.tensor(cfg.policy.latent_recon_threshold, dtype=torch.float32)
        threshold_source = "manual"

    # is_valid 是最后统一生成的硬过滤结果；训练时 OfflineDSRLLatentDataset 只保留 True 样本。
    cache_tensors["is_valid"] = recon_errors <= resolved_threshold

    recon_error_quantiles = {
        "recon_error_min": recon_errors.min(),
        "recon_error_p50": torch.quantile(recon_errors, 0.50),
        "recon_error_p75": torch.quantile(recon_errors, 0.75),
        "recon_error_p90": torch.quantile(recon_errors, 0.90),
        "recon_error_p95": torch.quantile(recon_errors, 0.95),
        "recon_error_p99": torch.quantile(recon_errors, 0.99),
        "recon_error_max": recon_errors.max(),
    }

    # 只用 valid 样本统计 return，AWR fallback 和日志才不会被无效反演样本污染。
    valid_returns = cache_tensors["return_to_go"][cache_tensors["is_valid"]]
    if valid_returns.numel() == 0:
        return_mean = torch.tensor(0.0)
        return_std = torch.tensor(1.0)
    else:
        return_mean = valid_returns.mean()
        return_std = valid_returns.std(unbiased=False).clamp_min(1e-6)

    # metadata 保存数据来源和关键统计，训练阶段不用重新扫描原始数据来恢复这些信息。
    metadata = {
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": str(cfg.dataset.root) if cfg.dataset.root is not None else None,
        "dataset_revision": cfg.dataset.revision,
        "query_stride": query_stride,
        "base_policy_path": str(cfg.policy.base_policy_path) if cfg.policy.base_policy_path is not None else None,
        "discount": cfg.policy.discount,
        "latent_inversion_steps": cfg.policy.latent_inversion_steps,
        "latent_inversion_patience": cfg.policy.latent_inversion_patience,
        "latent_inversion_min_delta": cfg.policy.latent_inversion_min_delta,
        "latent_inversion_decode_steps": cfg.policy.latent_inversion_decode_steps,
        "latent_inversion_log_freq": cfg.policy.latent_inversion_log_freq,
        "latent_recon_threshold": float(resolved_threshold.item()),
        "latent_recon_threshold_source": threshold_source,
        "latent_recon_quantile": cfg.policy.latent_recon_quantile,
        "latent_recon_quantile_threshold": float(quantile_threshold.item()),
        "latent_recon_max_threshold": cfg.policy.latent_recon_max_threshold,
        "num_total_samples": int(cache_tensors["dataset_index"].numel()),
        "num_valid_samples": int(cache_tensors["is_valid"].sum().item()),
        "valid_ratio": float(cache_tensors["is_valid"].to(dtype=torch.float32).mean().item()),
        "return_to_go_mean": float(return_mean.item()),
        "return_to_go_std": float(return_std.item()),
    }
    metadata.update({key: float(value.item()) for key, value in recon_error_quantiles.items()})

    # safetensors 写 tensor，JSON 写可读元信息；训练脚本只依赖这两个文件。
    save_dsrl_offline_cache(cfg.output_dir, cache_tensors, metadata)
    if tb_writer is not None:
        # 末尾再写一次总量指标，方便 TensorBoard 上快速确认 cache 规模。
        tb_writer.add_scalar("precompute/num_total_samples", metadata["num_total_samples"], offset)
        tb_writer.add_scalar("precompute/num_valid_samples", metadata["num_valid_samples"], offset)
        tb_writer.add_scalar("precompute/valid_ratio", metadata["valid_ratio"], offset)
        tb_writer.add_scalar("precompute/latent_recon_threshold", metadata["latent_recon_threshold"], offset)
        tb_writer.add_scalar("precompute/recon_error_p95", metadata["recon_error_p95"], offset)
        tb_writer.add_scalar("precompute/return_to_go_mean", metadata["return_to_go_mean"], offset)
        tb_writer.add_scalar("precompute/return_to_go_std", metadata["return_to_go_std"], offset)
        tb_writer.close()
    logging.info("Saved offline DSRL cache to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
