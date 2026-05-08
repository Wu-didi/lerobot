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
import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


# import debugpy
# debugpy.listen(12345)
# print("wait debug")
# debugpy.wait_for_client()
# print("Debugger attached")


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    执行一次完整的参数更新。

    这个函数是训练循环里真正“做反向传播”的核心步骤。它负责：
    1. 把 policy 切到 train 模式。
    2. 视情况计算 RA-BC 的样本权重。
    3. 在 accelerator.autocast() 下执行前向，得到 loss。
    4. 反向传播、梯度裁剪、optimizer.step()、scheduler.step()。
    5. 把本 step 的 loss / grad_norm / lr / update 耗时写回指标对象。

    这里故意把“单步更新”单独抽成函数，而不是全部塞进 train()：
    - train() 负责流程编排：取 batch、日志、保存、评估。
    - update_policy() 只负责“给定一个 batch，如何更新一次模型”。
    这样读源码时，训练主循环会更清楚。

    Args:
        train_metrics: 训练指标跟踪器。这个对象会在函数内部被原地更新。
        policy: 当前要训练的策略模型。
        batch: 经过 dataloader 取出、并且通常已经过 preprocessor 处理的一批数据。
        optimizer: 优化器。
        grad_clip_norm: 梯度裁剪阈值。<=0 时不做有限阈值裁剪，只统计总范数。
        accelerator: accelerate 的统一封装，负责分布式、混精、反向传播等。
        lr_scheduler: 学习率调度器，可选。
        lock: 可选锁。当前文件里默认不用，但保留接口给更特殊的并发更新场景。
        rabc_weights_provider: RA-BC 权重提供器。开启后会把 batch 内样本做加权。

    Returns:
        返回二元组：
        - 更新后的 train_metrics
        - policy 前向返回的 output_dict，主要给日志系统/W&B 继续记录
    """
    start_time = time.perf_counter()
    policy.train()

    # RA-BC 会根据 batch 中样本的进度/难度为每个样本生成一个权重。
    # 如果没有启用，就保持 None，后面走普通平均 loss 的分支。
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # 所有前向都放在 accelerator.autocast() 里，让 accelerate 决定是否启用混精。
    with accelerator.autocast():
        # 开启 RA-BC 时，不能直接拿“已经聚合好的平均 loss”。
        # 必须让 policy 返回每个样本各自的 loss，再按权重重算加权平均。
        if rabc_batch_weights is not None:
            # reduction="none" 代表保留每个样本单独的损失。
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # RA-BC 的核心公式：
            #   L = Σ(w_i * l_i) / (Σw_i + ε)
            # 这里额外加 epsilon 只是为了数值稳定，避免极端情况下分母为 0。
            # rabc_batch_weights 在 provider 内部通常已经做过归一化。
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # 下面这些额外统计值不会参与训练，只是为了日志里能看到
            # 当前 batch 的样本权重分布情况。
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            # 普通训练分支：policy 自己返回已经聚合好的 loss。
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # 统一通过 accelerator.backward() 做反向传播，这样在单卡/多卡/混精下
    # 都走同一套接口。
    accelerator.backward(loss)

    # 梯度裁剪是训练稳定性的常见手段。
    # grad_clip_norm > 0 时按给定阈值裁剪；
    # 否则不裁剪，但仍然统计一个“无限阈值”下的总梯度范数，便于日志观测。
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # 先 step，再 zero_grad，是 PyTorch 里最常见的更新顺序。
    # lock 只是在需要线程安全时才生效，这个训练脚本默认不会传。
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # 这个训练脚本的 scheduler 是“按 batch / step 更新”，不是按 epoch 更新。
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 某些 policy 除了参数外，还有内部缓存/统计量需要在 step 后刷新。
    # 如果模型实现了 update()，这里会显式调用一次。
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    # 把本 step 的关键训练指标回填到 tracker。
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    训练脚本主入口。

    这个函数负责把“离线训练”完整串起来，核心流程是：
    1. 校验配置，并构造 Accelerator。
    2. 初始化日志、随机种子、设备。
    3. 构造 dataset、可选 eval 环境、policy、processor、optimizer、scheduler。
    4. 如有需要，从 checkpoint 恢复训练状态。
    5. 构造 dataloader，进入主训练循环。
    6. 周期性做日志输出、存 checkpoint、跑评估。
    7. 训练结束后清理资源，并按配置把模型推到 Hub。

    读这个文件时，可以把 train() 当成“总调度器”：
    - 它不关心具体某个 policy 的内部细节。
    - 它关心的是训练系统层面的编排：谁先初始化、谁只在主进程执行、
      什么时候同步、什么时候保存和评估。

    Args:
        cfg: 完整训练配置。
        accelerator: 可选的 Accelerator 实例；为空时函数内部自动创建。
    """
    cfg.validate()

    # 如果外部没有传 accelerator，就在这里创建。
    # 这是整个脚本和 accelerate 集成的入口，后面的 device、DDP、混精、barrier
    # 都依赖这个对象。
    #
    # 这里有两个细节：
    # 1. step_scheduler_with_optimizer=False
    #    代表学习率调度完全由本脚本自己控制，不让 accelerate 自动改 step 逻辑。
    # 2. find_unused_parameters=True
    #    允许 DDP 容忍某些分支条件下未参与计算图的参数，适配条件计算模型。
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # accelerate 默认会根据硬件自动选设备。
        # 但如果配置里明确要求 policy.device == "cpu"，这里强制只用 CPU。
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # 多进程训练时，很多事情只应该由主进程做一次：
    # 例如打印日志、初始化 wandb、保存 checkpoint、创建 eval 环境等。
    is_main_process = accelerator.is_main_process

    # 打印完整配置通常只保留主进程一份，避免终端刷屏。
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # WandB 也只在主进程初始化，否则会创建重复 run。
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # 后续所有张量和模型都应该围绕 accelerator.device 工作，
    # 而不是自行猜测当前设备。
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 数据集的创建通常可能涉及下载、索引构建、缓存写入等副作用。
    # 为避免多进程同时创建产生竞争，先让主进程单独做一遍。
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    # 主进程把数据准备好后，其余进程再继续。
    accelerator.wait_for_everyone()

    # 此时其他进程再创建 dataset，就不会和主进程抢相同资源。
    if not is_main_process:
        dataset = make_dataset(cfg)

    # 训练过程中的在线评估环境只在以下条件下创建：
    # - 配置了 eval_freq
    # - 训练配置里提供了 env
    # - 当前是主进程
    #
    # 对真实机器人数据，通常不会在 train.py 里直接起环境评估；
    # 这部分一般由独立 eval 脚本完成。
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    # policy 的构造依赖 dataset meta：
    # 例如特征维度、统计量、episode 信息等。
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # CLI 传进来的 dataclass 配置要先转成 dict，才能作为覆盖项传给 policy。
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # 某些后续逻辑会假设所有进程上的模型结构都已经构造完毕，因此这里同步一次。
    accelerator.wait_for_everyone()

    # processor 负责把 dataset batch 转成 policy 真正需要的输入格式，
    # 以及把 policy 输出再映射回规范化/反规范化后的格式。
    #
    # 这里要特别注意“是否恢复训练”的区别：
    # - 如果是从已有 checkpoint/processor 状态恢复，尽量沿用保存下来的 processor 状态。
    # - 如果是新训练，或者只是从预训练模型初始化，则使用当前 dataset 的统计量构造 processor。
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # dataset_stats 决定了标准化/反标准化的尺度。
        # 恢复训练时不在这里强灌一份新的 stats，是为了避免覆盖 checkpoint 里保存的处理器状态。
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # SARM 除了统计量，还需要 dataset_meta 来处理 progress 相关的归一化。
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        # 从 pretrained_path 启动时，这里把和当前训练数据集相关的覆盖项传给 processor。
        # 这样既能复用预训练模型结构，又能让输入/输出映射到当前数据集的字段和统计量上。
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    # optimizer / scheduler 的具体类型由配置决定，这里只做工厂调用。
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # RA-BC 是一个可选训练增强逻辑。
    # 它需要预先计算好的 progress 信息，训练时再把这个 progress 映射成 batch 权重。
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # RA-BC 依赖 chunk_size，因为它通常和 chunk/prediction horizon 的定义绑定。
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    # step 表示已经做了多少次“参数更新”。
    # 这里的定义不是 epoch，也不是看过多少条样本，而是 forward + backward + optimizer.step 的次数。
    step = 0

    if cfg.resume:
        # 恢复训练时，除了 step，还需要把 optimizer / scheduler 一起恢复，
        # 否则学习率曲线和动量状态都会错位。
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        # 下面这一组日志主要用于开训前做 sanity check：
        # 看输出目录、环境任务、数据规模、有效 batch size、参数量是否符合预期。
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # 离线训练 dataloader：
    # 某些 policy 会要求“一个 episode 的最后 N 帧不能被采样”，
    # 例如因为要预测未来动作 chunk，尾部样本缺少足够上下文。
    # 这时就改用 EpisodeAwareSampler，而不是普通 shuffle。
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # accelerate.prepare() 会根据当前运行模式，把对象包成对应形式：
    # - policy 可能被包装成 DDP / mixed precision 版本
    # - dataloader 可能被替换成分布式 sampler 驱动的版本
    # 后面凡是要拿回原始模型，都必须用 accelerator.unwrap_model()。
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    # cycle() 把 dataloader 变成一个“无限迭代器”。
    # 这样训练循环只关心 step 数，不关心 epoch 边界。
    dl_iter = cycle(dataloader)

    policy.train()

    # AverageMeter 负责维护滑动统计；MetricsTracker 负责把这些统计组织成统一日志格式。
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        # 进度条只放在主进程显示，避免多进程互相覆盖终端输出。
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    # 主训练循环：
    # 每一轮做一件事：取一个 batch -> 预处理 -> 更新 policy -> 决定是否日志/保存/评估。
    for _ in range(step, cfg.steps):  # cfg.steps 是总更新步数，不是 epoch 数。
        start_time = time.perf_counter()
        batch = next(dl_iter)

        # preprocessor 负责字段改名、设备搬运、归一化等训练前准备。
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # 这里先把 step +1，再判断是否保存/评估。
        # 也就是说：
        # - “step = 1000 的 checkpoint”
        # - “step = 1000 的 eval”
        # 都表示“第 1000 次参数更新已经完成之后”的状态。
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    # 模型 forward 里返回的额外监控指标，例如各类辅助 loss、
                    # RA-BC 统计等，也合并进 wandb 日志。
                    wandb_log_dict.update(output_dict)
                # RA-BC 还有一组全局统计值，和当前 batch 的 output_dict 不完全重复。
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)

            # 一次日志输出之后，把 AverageMeter 的累计窗口清掉，
            # 下一个日志周期重新开始统计平均值。
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # 保存完以后做一次 barrier，确保别的进程不会在主进程还没写完 checkpoint 时继续往前跑。
            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")

                # 评估不需要梯度，但仍然允许 autocast，这样可以复用推理时的混精收益。
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )

                # overall 是跨 suite 聚合后的总指标，最适合做横向比较。
                aggregated = eval_info["overall"]

                # 除了 overall，也保留每个 suite 的聚合结果，方便看不同任务组表现。
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # 评估也复用 MetricsTracker，保持训练/评估两套日志格式尽量一致。
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    # 默认把 overall 的第一段视频同步到 wandb，便于快速人工查看策略表现。
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            # 评估结束后同步，避免主进程长时间 eval、其他进程已经继续训练导致步数错位。
            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            # PEFT 模型和普通模型推送 Hub 的接口不完全一样，所以分开处理。
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # 训练结束前再做一次 barrier，确保所有进程都走到收尾阶段，再统一释放 accelerate 资源。
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    """CLI 入口：先注册第三方插件，再进入 train()。"""
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
