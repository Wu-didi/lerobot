#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    """
    Configuration for the PI0.5 policy.
    PI0.5 策略的配置对象。

    This dataclass defines the model architecture, training hyperparameters,
    normalization behavior, and RTC-related options used by the pi05 policy.
    这个 dataclass 统一描述 pi05 使用的模型结构、训练超参数、归一化行为，
    以及和 RTC / training-time RTC 相关的选项。
    """

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    # --- Training-Time RTC (arXiv 2512.05964) ---
    # 普通 action-chunk policy 训练时默认“整段 chunk 都是未知的”，模型从噪声生成完整动作块。
    # 但真实 RTC 部署时，新的 chunk 生成出来之前，机器人已经在执行上一段 chunk；
    # 下一段 chunk 的前几个动作往往已经被上一段 chunk 的剩余动作决定。
    # training_rtc=True 时，训练阶段就模拟这个结构：
    #   - chunk 前面一小段 token 被当作已知 clean prefix；
    #   - 模型只需要在这个 prefix 条件下补全后面的 suffix；
    #   - 这样推理时可以直接把上一块剩余动作塞进当前 chunk，而不是完全依赖 VJP/pinv guidance。
    training_rtc: bool = False  # Enable training-time action conditioning for real-time chunking

    # simulated_delay 是训练时最多模拟多少个“已确定前缀”token。
    # 实际每个样本会从 {0, ..., simulated_delay - 1} 里采一个 delay；
    # delay=0 保留普通 flow-matching 样本，delay>0 则表示前 delay 个 token 是 frozen prefix。
    # 这里和官方 kinetix 实现一致使用指数权重，让较小 delay 更常见，贴近实际推理延迟通常较短的分布。
    simulated_delay: int = 5  # Max prefix delay K; delay sampled from {0,...,K-1} with exp weights

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        """
        Run post-initialization validation.
        执行配置的后置校验。

        This hook ensures mutually dependent hyperparameters are consistent
        before the rest of the policy stack starts using the config.
        这个钩子会在配置真正进入模型与处理器之前，先检查关键超参数是否互相兼容。
        """
        super().__post_init__()

        # Validate configuration
        # 校验 action horizon / chunk horizon 的基本关系，避免运行时维度不一致。
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.training_rtc:
            if self.simulated_delay <= 0 or self.simulated_delay > self.chunk_size:
                raise ValueError(
                    f"simulated_delay must be in [1, {self.chunk_size}], got {self.simulated_delay}"
                )

    def validate_features(self) -> None:
        """
        Validate and set up input/output features.
        校验并补齐输入/输出特征定义。

        PI0.5 expects image, state, and action feature specs to exist.
        If the caller did not provide them explicitly, this method inserts the
        required default placeholders based on config dimensions.
        PI0.5 期望输入里至少有图像/状态定义，输出里至少有动作定义。
        如果外部没有显式给出，这里会按配置里的维度补出默认定义。
        """
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        """
        Return the optimizer preset used by PI0.5.
        返回 PI0.5 默认使用的优化器配置。
        """
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """
        Return the scheduler preset used by PI0.5.
        返回 PI0.5 默认使用的学习率调度器配置。
        """
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """
        PI0.5 does not request stacked observation windows via dataset deltas.
        PI0.5 默认不通过 dataset delta 机制额外堆叠 observation 时间窗口。
        """
        return None

    @property
    def action_delta_indices(self) -> list:
        """
        Return the action horizon as delta indices.
        以 delta index 形式返回动作 horizon。

        During training, the dataset should expose the entire action chunk
        ``[0, 1, ..., chunk_size - 1]`` so the policy can supervise the full horizon.
        训练时数据集需要把完整动作块都暴露出来，模型才能监督整个 action chunk。
        """
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """
        PI0.5 does not consume reward deltas from the dataset.
        PI0.5 默认不读取 reward 的时序窗口。
        """
        return None
