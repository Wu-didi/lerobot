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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("dsrl_pi05")
@dataclass
class DSRLPi05Config(PI05Config):
    """pi0.5 之上的离线 DSRL 风格 latent-noise 后训练配置。

    这个配置继承 `PI05Config`，原因是 DSRL 仍然要复用 pi0.5 的图像、语言、
    state、action chunk 等输入输出定义。新增字段只描述 DSRL 自己的 latent
    actor / critic / value，以及 action-to-noise inversion 和 offline RL 超参。
    """

    # 冻结的 base pi0.5 checkpoint 路径。DSRL 只训练外部 latent 头，base policy 作为解码器使用。
    base_policy_path: Path | None = None

    # 观测表征维度。默认从 pi0.5 的 PaliGemma hidden size 推断，只有推断失败时才需要手动指定。
    observation_feature_dim: int | None = None
    # actor 输入观测表征，输出 flattened latent noise，因此只需要 MLP，不需要再训练视觉语言主干。
    policy_hidden_dims: list[int] = field(default_factory=lambda: [1024, 1024])
    # twin critic 分别估计 Q1(s, z)、Q2(s, z)，用两个网络降低 Q 过估计风险。
    critic_hidden_dims: list[int] = field(default_factory=lambda: [1024, 1024])
    # value head 估计 V(s)，IQL 需要单独的 state value 来做 expectile regression。
    value_hidden_dims: list[int] = field(default_factory=lambda: [1024, 1024])

    # 第一版默认用 IQL；保留 AWR 是为了能退化成更简单的 return-weighted BC 对照实验。
    offline_algorithm: str = "iql"
    # query_stride 控制每隔多少原始 frame 取一个 chunk-level transition；默认等于 n_action_steps。
    query_stride: int | None = None
    # sparse reward 的折扣因子，用于 Bellman target 和 return-to-go 统计。
    discount: float = 0.99
    # AWR 权重温度。值越大，高 return 样本权重越集中。
    awr_beta: float = 1.0
    # AWR 权重裁剪，避免极端 return 让少数样本主导训练。
    awr_weight_clip: float = 20.0
    # IQL value expectile。大于 0.5 会让 V 更偏向高 Q 区域，这是 IQL 的 policy improvement 信号来源。
    iql_expectile: float = 0.7
    # IQL actor advantage 权重温度。越大越偏向拟合高 advantage 的 latent action。
    iql_temperature: float = 3.0
    # advantage 权重上限，防止 exp(temperature * advantage) 数值爆炸。
    iql_adv_clip: float = 100.0

    # latent noise 的可选 clamp 幅度。None 表示完全不裁剪，和 pi0.5 inference 的 sample_noise 行为一致。
    # DSRL 应该默认工作在 pi0.5 原始高斯 noise space；只有做安全消融时才显式设置 clamp。
    noise_action_magnitude: float | None = None

    # 每个样本反演 latent noise 的最大 Adam 优化步数，也就是反演的 max_steps。
    # 实际运行时如果 patience 早停触发，会提前结束；否则最多跑这么多步。
    latent_inversion_steps: int = 200
    # 反演早停 patience。连续这么多步重建误差 improvement 小于 min_delta 就停止当前 restart。
    # 这样已经收敛的样本不会继续浪费 pi0.5 decoder 计算；设为 None 可关闭早停。
    latent_inversion_patience: int | None = 30
    # 反演早停的最小改善量。小于这个值视为没有有效进步。
    latent_inversion_min_delta: float = 1e-5
    # 反演时 pi0.5 flow decoder 使用的 denoise 步数。None 表示沿用 num_inference_steps。
    # 反演阶段每个 Adam step 都要跑一次 decoder，所以这里通常应小于正式推理步数，用速度换可接受的 label 精度。
    latent_inversion_decode_steps: int | None = None
    # 反演内层优化日志频率。0 表示关闭；大于 0 时每隔这么多 Adam step 打印一次 recon_error/loss。
    latent_inversion_log_freq: int = 0
    # 多次随机初始化反演，保留重建误差最小的 noise，减少局部最优影响。
    latent_restarts: int = 4
    # 反演时只优化 latent noise，因此学习率可以比普通模型训练高。
    latent_inversion_lr: float = 5e-2
    # L2 正则约束反演出的 noise 不要过大，避免靠异常 latent 勉强重建 demo action。
    latent_reg_weight: float = 1e-4
    # 手动重建误差阈值。None 表示不提前指定阈值，而是在 cache 全部生成后根据 recon_error 分布自动求阈值。
    # 这样可以先看完整数据分布，再统一决定哪些 noise_label 可信。
    latent_recon_threshold: float | None = None
    # 自动阈值使用的 recon_error 分位数。0.95 表示保留重建误差最低的约 0.95 比例样本。
    latent_recon_quantile: float = 0.95
    # 自动阈值的可选绝对上限。None 表示只使用分位数；设置后 threshold = min(quantile_threshold, max_threshold)。
    latent_recon_max_threshold: float | None = None
    # 是否按 recon_error 对 actor 训练样本再做质量加权。默认关闭，保持第一版训练目标简单。
    # 开启后 recon_error 越小权重越大，recon_error 接近阈值的样本权重会降低。
    recon_error_weighting: bool = False
    # recon_error 质量权重温度：weight = exp(-recon_error / temperature)。
    # 值越小，训练越偏向重建误差很低的 latent label。
    recon_error_weight_temperature: float = 5e-2

    # 这些 optimizer 字段保留 LeRobot policy 统一接口；实际 IQL 脚本会分别创建 actor/critic/value optimizer。
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0
    # actor/critic/value 分开设学习率，因为 critic/value 往往需要比 actor 更快拟合离线 Bellman 目标。
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4

    def __post_init__(self) -> None:
        """在 pi0.5 基础配置校验之后，再校验 DSRL 自己的超参数。

        这里先调用 `PI05Config.__post_init__()`，确保 pi0.5 的 feature、chunk、
        action dim 等基础配置合法；然后再检查 DSRL 新增字段，尽早把错误配置挡在训练前。
        """
        super().__post_init__()

        # 只允许当前实现真正支持的离线算法，避免用户传入 sac/cql 等还未实现的名字后静默跑错。
        if self.offline_algorithm not in {"awr", "iql"}:
            raise ValueError(
                f"Unsupported offline_algorithm='{self.offline_algorithm}'. Supported values: 'awr', 'iql'."
            )
        # 反演至少要有一步优化，否则无法从 demo action 得到 latent noise label。
        if self.latent_inversion_steps <= 0:
            raise ValueError("latent_inversion_steps must be > 0")
        # patience 为 None 表示关闭早停；否则必须为正数，0 会让第一步之后就可能错误停止。
        if self.latent_inversion_patience is not None and self.latent_inversion_patience <= 0:
            raise ValueError("latent_inversion_patience must be > 0 or None")
        # min_delta 允许为 0，表示只要误差没有严格下降就累计 patience；负数没有语义。
        if self.latent_inversion_min_delta < 0:
            raise ValueError("latent_inversion_min_delta must be >= 0")
        # decode_steps 为 None 时使用 pi0.5 默认推理步数；显式设置时必须为正。
        if self.latent_inversion_decode_steps is not None and self.latent_inversion_decode_steps <= 0:
            raise ValueError("latent_inversion_decode_steps must be > 0 or None")
        # log_freq=0 表示关闭日志；负数没有语义。
        if self.latent_inversion_log_freq < 0:
            raise ValueError("latent_inversion_log_freq must be >= 0")
        # restart 数必须为正，否则没有候选 latent 可以被评估和保存。
        if self.latent_restarts <= 0:
            raise ValueError("latent_restarts must be > 0")
        # Adam 学习率必须为正，0 或负值会让反演无效或直接报错。
        if self.latent_inversion_lr <= 0:
            raise ValueError("latent_inversion_lr must be > 0")
        # discount 允许等于 1，但不能超过 1；超过 1 会让长 episode 回报发散。
        if self.discount <= 0 or self.discount > 1:
            raise ValueError("discount must be in (0, 1]")
        # AWR 权重下限固定为 1，因此 clip 小于 1 没有意义。
        if self.awr_weight_clip < 1.0:
            raise ValueError("awr_weight_clip must be >= 1.0")
        # None 表示关闭 clamp；显式设置时必须为正，否则会把 noise 裁成无效范围。
        if self.noise_action_magnitude is not None and self.noise_action_magnitude <= 0:
            raise ValueError("noise_action_magnitude must be > 0 or None")
        # expectile 是分位型回归参数，只在 (0, 1) 内有定义。
        if not 0 < self.iql_expectile < 1:
            raise ValueError("iql_expectile must be in (0, 1)")
        # IQL temperature 是 exp 权重的缩放系数，必须为正。
        if self.iql_temperature <= 0:
            raise ValueError("iql_temperature must be > 0")
        # advantage clip 必须为正，否则 actor 权重会被错误裁成非正数。
        if self.iql_adv_clip <= 0:
            raise ValueError("iql_adv_clip must be > 0")
        # 手动阈值为 None 时走自动分位数；如果用户显式给阈值，则必须为正。
        if self.latent_recon_threshold is not None and self.latent_recon_threshold <= 0:
            raise ValueError("latent_recon_threshold must be > 0 or None")
        # 分位数必须在 (0, 1] 内；1.0 表示只过滤超过最大误差上限的样本。
        if not 0 < self.latent_recon_quantile <= 1:
            raise ValueError("latent_recon_quantile must be in (0, 1]")
        # 自动阈值上限为 None 时关闭；显式设置时必须为正。
        if self.latent_recon_max_threshold is not None and self.latent_recon_max_threshold <= 0:
            raise ValueError("latent_recon_max_threshold must be > 0 or None")
        # recon_error 质量权重使用 exp(-error / temperature)，temperature 必须为正。
        if self.recon_error_weight_temperature <= 0:
            raise ValueError("recon_error_weight_temperature must be > 0")
        # 三个 optimizer 都实际参与训练，所以都要在配置阶段检查。
        if self.actor_lr <= 0 or self.critic_lr <= 0 or self.value_lr <= 0:
            raise ValueError("actor_lr, critic_lr, and value_lr must be > 0")

    def get_optimizer_preset(self) -> AdamWConfig:
        """返回一个兼容 LeRobot 通用接口的 AdamW optimizer 配置。

        这个方法主要用于满足 `PreTrainedConfig` 的统一接口。离线 IQL 训练脚本
        会手动创建三个 optimizer，但保留这个 preset 可以让 policy factory 和测试代码
        按普通 policy 的方式工作。
        """
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """默认关闭 scheduler，让独立离线 DSRL 脚本只关注 RL loss 本身。

        第一版离线 DSRL 训练目标是验证 latent RL 链路，先不引入 scheduler，
        避免把 loss 变化和学习率变化混在一起。
        """
        return None

    @property
    def latent_action_dim(self) -> int:
        """返回 actor 和 critic 使用的展平 latent action 维度。

        pi0.5 解码的是 `[chunk_size, max_action_dim]` 的 noise chunk；actor/critic
        的 MLP 更方便处理一维向量，所以这里统一提供展平后的维度。
        """
        return self.chunk_size * self.max_action_dim
