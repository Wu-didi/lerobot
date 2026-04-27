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

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dsrl_pi05.configuration_dsrl_pi05 import DSRLPi05Config
from lerobot.policies.dsrl_pi05.offline_utils import (
    OfflineDSRLLatentDataset,
    build_awr_weights,
    load_dsrl_offline_cache,
    make_tensorboard_writer,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class DSRLOfflineTrainConfig:
    """离线 DSRL / latent IQL 训练脚本配置。"""

    # DSRLPi05Config 包含 base pi0.5 路径和 actor/critic/value 超参。
    policy: DSRLPi05Config
    # precompute 脚本输出的 cache 目录，必须包含 safetensors 和 JSON metadata。
    cache_dir: Path
    # checkpoint、training_state 和 TensorBoard 日志输出目录。
    output_dir: Path = Path("outputs/dsrl/offline_train")
    # 离线 RL 训练 batch size；显存主要消耗在冻结 pi0.5 编码观测。
    batch_size: int = 8
    # 默认 0 便于真实数据调试；需要更高吞吐时可以调大。
    num_workers: int = 0
    # 训练总 step 数，不按 epoch 结束，因为 offline cache 可被重复采样。
    steps: int = 10_000
    # 控制 tqdm/logging 刷新频率。
    log_freq: int = 50
    # 控制中间 checkpoint 保存频率。
    save_freq: int = 1_000
    # 固定随机种子，保证 dataloader shuffle 和模型初始化可复现。
    seed: int = 0
    # 是否写 TensorBoard scalar。
    tensorboard: bool = True
    # 可选 TensorBoard 目录；为空时写到 output_dir/runs/train。
    tensorboard_log_dir: Path | None = None


def make_lerobot_dataset_from_cache(
    cache_metadata: dict, policy_cfg: DSRLPi05Config
) -> LeRobotDataset:
    """根据 cache metadata 恢复原始 LeRobotDataset。

    cache 只保存索引和 latent label，不复制原始图像/state；
    因此训练时需要回到同一个 dataset root，用 cache 里的 dataset_index 取样本。
    """
    dataset_root = cache_metadata.get("dataset_root")
    # metadata 里记录了 repo_id/root/revision，保证训练读取的是生成 cache 时同一份数据。
    ds_meta = LeRobotDatasetMetadata(
        cache_metadata["dataset_repo_id"],
        root=dataset_root,
        revision=cache_metadata.get("dataset_revision"),
    )
    # policy 的 delta_timestamps 必须与 precompute 阶段一致，否则 dataset_index 对应的 action chunk 会错位。
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    return LeRobotDataset(
        cache_metadata["dataset_repo_id"],
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        revision=cache_metadata.get("dataset_revision"),
    )


def collate_offline_dsrl_batch(samples):
    """把 OfflineDSRLLatentDataset 的三元组样本合成 batch。

    每个样本包含当前 raw obs、下一步 raw obs、以及 cache labels；
    分开 collate 可以让 preprocessor 分别处理 s 和 s'。
    """
    raw_items, next_raw_items, labels = zip(*samples, strict=True)
    return default_collate(list(raw_items)), default_collate(list(next_raw_items)), default_collate(list(labels))


def make_dsrl_preprocessor(policy, dataset_stats):
    """构建和 base pi0.5 一致的 processor。

    离线训练阶段仍然需要把原始 LeRobot 样本转换成 pi0.5 输入格式；
    processor 必须和 precompute 阶段一致，否则 latent label 和观测表征会错位。
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
def main(cfg: DSRLOfflineTrainConfig) -> None:
    """运行离线 DSRL 训练。

    默认算法是 IQL：先训练 critic 拟合 sparse reward Bellman target，再训练 value 做
    expectile regression，最后用 advantage-weighted BC 更新 latent actor。
    """
    init_logging()
    set_seed(cfg.seed)

    # 读取预计算 cache；这里不会重新执行昂贵的 action-to-noise inversion。
    cache_tensors, cache_metadata = load_dsrl_offline_cache(cfg.cache_dir)
    # 用 cache metadata 恢复原始数据集，训练时按 cache 里的 dataset_index 取样本。
    dataset = make_lerobot_dataset_from_cache(cache_metadata, cfg.policy)
    # make_policy 会实例化 DSRLPi05Policy，并加载冻结 base pi0.5。
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    # preprocessor 把 raw obs/action 转成 pi0.5/DSRL policy 可消费的 tensor。
    preprocessor, _ = make_dsrl_preprocessor(policy, dataset.meta.stats)
    # AWR fallback 需要 return 标准化统计；IQL 路径保留这些 buffer 也不影响训练。
    policy.set_return_to_go_stats(
        cache_metadata.get("return_to_go_mean", 0.0),
        cache_metadata.get("return_to_go_std", 1.0),
    )

    # OfflineDSRLLatentDataset 会自动过滤 is_valid=False 的反演失败样本。
    offline_dataset = OfflineDSRLLatentDataset(dataset, cache_tensors)
    dataloader = DataLoader(
        offline_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_offline_dsrl_batch,
        drop_last=False,
    )
    # actor、critic、value 分别优化，方便按 IQL 的三类 loss 独立调学习率和梯度裁剪。
    actor_optimizer = torch.optim.AdamW(
        policy.actor.parameters(),
        lr=policy.config.actor_lr,
        betas=policy.config.optimizer_betas,
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        list(policy.critic1.parameters()) + list(policy.critic2.parameters()),
        lr=policy.config.critic_lr,
        betas=policy.config.optimizer_betas,
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )
    value_optimizer = torch.optim.AdamW(
        policy.value_head.parameters(),
        lr=policy.config.value_lr,
        betas=policy.config.optimizer_betas,
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )

    # 输出目录用于保存中间 checkpoint、final checkpoint 和 TensorBoard event。
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = None
    if cfg.tensorboard:
        tb_writer = make_tensorboard_writer(cfg.tensorboard_log_dir or cfg.output_dir / "runs" / "train")

    # 只训练 DSRL head；base pi0.5 参数已经 requires_grad=False。
    policy.train()

    progress = tqdm(total=cfg.steps, desc="Training offline DSRL")
    step = 0
    running_actor_loss = 0.0
    while step < cfg.steps:
        # dataloader 会反复遍历 cache；达到 steps 后手动停止。
        for raw_batch, next_raw_batch, labels in dataloader:
            if step >= cfg.steps:
                break

            # 当前状态和下一状态都要 processor，因为 critic target 使用 V(s')。
            processed_batch = preprocessor(raw_batch)
            next_processed_batch = preprocessor(next_raw_batch)
            # behavior_noise 是 demo action 反演得到的 latent action 标签。
            behavior_noise = labels["noise_label"].to(policy.config.device)
            # reward/done 来自 sparse reward table，是 RL 后训练信号。
            reward = labels["reward"].to(policy.config.device)
            done = labels["done"].to(policy.config.device, dtype=torch.float32)

            # 冻结 pi0.5 编码观测；后续三个 head 都复用同一份特征。
            obs_features = policy._encode_observation_context(processed_batch)
            next_obs_features = policy._encode_observation_context(next_processed_batch)

            # 第一步：critic 拟合 r + gamma * V(s')，学习 latent action 的长期价值。
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss, critic_dict = policy.compute_critic_loss(
                obs_features=obs_features,
                behavior_noise=behavior_noise,
                reward=reward,
                next_obs_features=next_obs_features,
                done=done,
            )
            critic_loss.backward()
            # 裁剪梯度，避免 sparse reward/初期 Q 值不稳定导致爆炸。
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(policy.critic1.parameters()) + list(policy.critic2.parameters()),
                policy.config.optimizer_grad_clip_norm,
            )
            critic_optimizer.step()

            # 第二步：value 用 expectile regression 拟合 dataset action 的 conservative Q。
            value_optimizer.zero_grad(set_to_none=True)
            value_loss, value_dict = policy.compute_value_loss(
                obs_features=obs_features,
                behavior_noise=behavior_noise,
            )
            value_loss.backward()
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.value_head.parameters(),
                policy.config.optimizer_grad_clip_norm,
            )
            value_optimizer.step()

            # 第三步：actor 只拟合离线数据里的 latent action，但按 IQL advantage 或 AWR return 加权。
            actor_optimizer.zero_grad(set_to_none=True)
            if policy.config.offline_algorithm == "iql":
                # IQL 路径：权重来自 Q(s,z)-V(s)，体现 policy improvement。
                actor_loss, actor_dict = policy.compute_iql_actor_loss(
                    obs_features=obs_features,
                    behavior_noise=behavior_noise,
                )
            else:
                # AWR fallback：不依赖 critic/value 的 advantage，直接用 return-to-go 加权 BC。
                sample_weight = build_awr_weights(
                    labels["return_to_go"].to(policy.config.device).reshape(-1),
                    beta=policy.config.awr_beta,
                    clip=policy.config.awr_weight_clip,
                    mean=policy.return_to_go_mean,
                    std=policy.return_to_go_std,
                )
                actor_loss, actor_dict = policy.compute_actor_loss_from_features(
                    obs_features=obs_features,
                    target_noise=behavior_noise,
                    sample_weight=sample_weight,
                )
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.actor.parameters(),
                policy.config.optimizer_grad_clip_norm,
            )
            actor_optimizer.step()

            step += 1
            running_actor_loss += actor_loss.item()
            progress.update(1)

            if tb_writer is not None:
                # TensorBoard 记录三类 loss、梯度范数和 sparse reward 统计，用于判断训练是否稳定。
                tb_writer.add_scalar("train/actor_loss", actor_loss.item(), step)
                tb_writer.add_scalar("train/critic_loss", critic_dict["critic_loss"], step)
                tb_writer.add_scalar("train/value_loss", value_dict["value_loss"], step)
                tb_writer.add_scalar("train/mse_loss", actor_dict["mse_loss"], step)
                tb_writer.add_scalar("train/mean_weight", actor_dict["mean_weight"], step)
                tb_writer.add_scalar("train/actor_grad_norm", actor_grad_norm.item(), step)
                tb_writer.add_scalar("train/critic_grad_norm", critic_grad_norm.item(), step)
                tb_writer.add_scalar("train/value_grad_norm", value_grad_norm.item(), step)
                tb_writer.add_scalar("train/reward_mean", reward.mean().item(), step)
                tb_writer.add_scalar("train/done_mean", done.mean().item(), step)

            if step % cfg.log_freq == 0 or step == 1:
                # tqdm/logging 只打印滑动 actor loss 和当前 critic/value，避免每步刷屏。
                avg_actor_loss = running_actor_loss / min(cfg.log_freq, step)
                running_actor_loss = 0.0
                progress.set_postfix(
                    actor=f"{avg_actor_loss:.6f}",
                    critic=f"{critic_dict['critic_loss']:.6f}",
                    value=f"{value_dict['value_loss']:.6f}",
                )
                logging.info(
                    "step=%s actor=%.6f critic=%.6f value=%.6f mse=%.6f mean_weight=%.4f "
                    "actor_grad=%.4f critic_grad=%.4f value_grad=%.4f",
                    step,
                    avg_actor_loss,
                    critic_dict["critic_loss"],
                    value_dict["value_loss"],
                    actor_dict["mse_loss"],
                    actor_dict["mean_weight"],
                    actor_grad_norm.item(),
                    critic_grad_norm.item(),
                    value_grad_norm.item(),
                )

            if step % cfg.save_freq == 0 or step == cfg.steps:
                # 中间 checkpoint 只保存 DSRL head 和 config；base pi0.5 由 base_policy_path 引用。
                checkpoint_dir = cfg.output_dir / f"step_{step:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                with open(checkpoint_dir / "training_state.json", "w") as f:
                    json.dump(
                        {
                            "step": step,
                            "cache_dir": str(cfg.cache_dir),
                            "return_to_go_mean": float(policy.return_to_go_mean.item()),
                            "return_to_go_std": float(policy.return_to_go_std.item()),
                        },
                        f,
                        indent=2,
                    )

    # 训练结束后额外保存 final 目录，方便部署/评估脚本使用固定路径。
    final_dir = cfg.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    with open(final_dir / "training_summary.json", "w") as f:
        # summary 记录训练步数和 cache 规模，方便之后追踪这次实验的来源。
        json.dump(
            {
                "steps": cfg.steps,
                "cache_dir": str(cfg.cache_dir),
                "num_valid_cache_samples": len(offline_dataset),
            },
            f,
            indent=2,
        )
    if tb_writer is not None:
        # 末尾写入有效样本数，TensorBoard 上可以直接检查 cache 过滤后的规模。
        tb_writer.add_scalar("train/num_valid_cache_samples", len(offline_dataset), cfg.steps)
        tb_writer.close()
    logging.info("Saved offline DSRL checkpoint to %s", final_dir)


if __name__ == "__main__":
    main()
