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

from collections import deque
from pathlib import Path
from typing import Unpack

import torch
import torch.nn.functional as F
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.dsrl_pi05.configuration_dsrl_pi05 import DSRLPi05Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def _safe_tqdm_write(message: str) -> None:
    """在 tqdm 进度条存在时安全打印一行日志。

    precompute 脚本外层已经有 tqdm；直接 print 会破坏进度条显示。
    这里优先用 tqdm.write，没有 tqdm 时再退回普通 print。
    """
    try:
        from tqdm import tqdm

        tqdm.write(message)
    except Exception:
        print(message)


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    """构建 actor、critic 或 value 使用的小型 MLP head。

    DSRL 第一版不训练 pi0.5 主体，只在冻结的观测表征上训练轻量头。
    因此这里统一用 MLP，避免为 actor/critic/value 写三套重复结构。
    """
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        # 每一段都是 Linear；中间层加 SiLU，最后一层保持线性以便输出任意 latent/Q/V 数值。
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _expectile_loss(diff: Tensor, expectile: float) -> Tensor:
    """计算 IQL value 使用的 expectile regression loss。

    IQL 不直接用 policy 采样做 Bellman backup，而是让 V(s) 用 expectile 拟合
    dataset action 的 Q(s, z)。当 diff=Q-V 为正时给 expectile 权重，否则给 1-expectile。
    """
    weight = torch.where(diff > 0, torch.full_like(diff, expectile), torch.full_like(diff, 1.0 - expectile))
    return weight * diff.square()


class DSRLPi05Policy(PreTrainedPolicy):
    """冻结 pi0.5 之上的离线 DSRL 风格 latent-noise policy。

    这个类的核心思想是：pi0.5 负责把 latent noise 解码成真实 action chunk，
    DSRL 负责学习“什么 noise 更好”。因此这里训练的是 actor/Q/V 外部头，
    而不是 full-parameter finetune pi0.5。
    """

    config_class = DSRLPi05Config
    name = "dsrl_pi05"

    def __init__(
        self,
        config: DSRLPi05Config,
        base_policy: PI05Policy | None = None,
        **kwargs,
    ):
        """初始化冻结的 base pi0.5，以及可训练的 latent RL head。

        `base_policy` 参数主要给测试使用；真实训练中会从 `base_policy_path`
        加载 pi0.5 checkpoint。加载后立刻冻结，确保 offline RL 不意外改动 base policy。
        """
        super().__init__(config)
        self.config = config

        # 测试里可能直接注入 fake base policy，此时继承它的 feature 定义，避免手动重复配置。
        if not self.config.input_features and base_policy is not None:
            self.config.input_features = dict(base_policy.config.input_features)
        if not self.config.output_features and base_policy is not None:
            self.config.output_features = dict(base_policy.config.output_features)
        # LeRobot policy 需要 input/output feature 合法，否则 processor 和 dataset 对不齐。
        self.config.validate_features()

        # base pi0.5 是 frozen decoder；actor 输出 noise 后交给它解码动作。
        self.base_policy = base_policy if base_policy is not None else self._load_base_policy()
        self._validate_base_policy_compatibility()

        # 明确冻结全部 base 参数，防止 optimizer 或 backward 意外更新 pi0.5 主体。
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        # base policy 只做表征和解码，不需要 dropout/training-mode 行为。
        self.base_policy.eval()

        # 观测表征维度通常等于 PaliGemma text hidden size。
        observation_feature_dim = self._infer_observation_feature_dim()
        # actor: s -> z，输出 flattened latent noise。
        self.actor = _build_mlp(
            input_dim=observation_feature_dim,
            hidden_dims=self.config.policy_hidden_dims,
            output_dim=self.config.latent_action_dim,
        )
        # critic 输入是 (s, z)，所以维度是观测表征加 flattened latent action。
        critic_input_dim = observation_feature_dim + self.config.latent_action_dim
        # twin Q 是 IQL/actor-critic 常用做法，训练 actor 时取 min(Q1,Q2) 降低过估计。
        self.critic1 = _build_mlp(
            input_dim=critic_input_dim,
            hidden_dims=self.config.critic_hidden_dims,
            output_dim=1,
        )
        self.critic2 = _build_mlp(
            input_dim=critic_input_dim,
            hidden_dims=self.config.critic_hidden_dims,
            output_dim=1,
        )
        self.value_head = _build_mlp(
            input_dim=observation_feature_dim,
            hidden_dims=self.config.value_hidden_dims,
            output_dim=1,
        )

        # AWR fallback 需要 return 标准化；保存成 buffer 让 checkpoint 里也带着这些统计。
        self.register_buffer("return_to_go_mean", torch.zeros(1), persistent=True)
        self.register_buffer("return_to_go_std", torch.ones(1), persistent=True)

        # 初始化动作队列，select_action 时按 n_action_steps 逐步吐出 action chunk。
        self.reset()

    def _load_base_policy(self) -> PI05Policy:
        """从磁盘加载将被冻结使用的 pi0.5 base policy。

        这里用 `PreTrainedConfig.from_pretrained` 而不是 `PI05Config.from_pretrained`，
        是因为真实保存的 config.json 带有 `type=pi05` 字段，需要通用 registry 先解析类型。
        """
        if self.config.base_policy_path is None:
            raise ValueError("base_policy_path is required to instantiate DSRLPi05Policy")

        base_cfg = PreTrainedConfig.from_pretrained(self.config.base_policy_path)
        if not isinstance(base_cfg, PI05Config):
            raise TypeError(
                f"Expected a pi0.5 config at {self.config.base_policy_path}, got {type(base_cfg).__name__} instead."
            )
        base_cfg.device = self.config.device
        return PI05Policy.from_pretrained(self.config.base_policy_path, config=base_cfg, strict=False)

    def _validate_base_policy_compatibility(self) -> None:
        """检查 DSRL 配置和 base pi0.5 的 action chunk 形状是否一致。

        actor/critic 的 latent 维度来自 DSRL config，而 base policy 解码动作来自 pi0.5 config；
        两者 chunk_size/max_action_dim/action feature 不一致会导致 silent shape bugs。
        """
        base_cfg = self.base_policy.config
        if base_cfg.chunk_size != self.config.chunk_size:
            raise ValueError(
                f"Base pi0.5 chunk_size ({base_cfg.chunk_size}) does not match DSRL config chunk_size ({self.config.chunk_size})"
            )
        if base_cfg.max_action_dim != self.config.max_action_dim:
            raise ValueError(
                f"Base pi0.5 max_action_dim ({base_cfg.max_action_dim}) does not match DSRL config max_action_dim ({self.config.max_action_dim})"
            )

        base_action_feature = base_cfg.output_features.get(ACTION)
        dsrl_action_feature = self.config.output_features.get(ACTION)
        if base_action_feature is not None and dsrl_action_feature is not None:
            if base_action_feature.shape != dsrl_action_feature.shape:
                raise ValueError(
                    f"Base pi0.5 action shape {base_action_feature.shape} does not match DSRL action shape {dsrl_action_feature.shape}"
                )

    def _infer_observation_feature_dim(self) -> int:
        """从冻结的 pi0.5 模型中推断 pooled observation feature 维度。

        DSRL heads 接收的是 prefix hidden state 的 pooled 表征。这个 hidden size 应该和
        PaliGemma text_config.hidden_size 一致；如果模型结构变了，就要求用户显式配置。
        """
        if self.config.observation_feature_dim is not None:
            return self.config.observation_feature_dim

        text_config = getattr(
            getattr(self.base_policy.model.paligemma_with_expert.paligemma.config, "text_config", None),
            "hidden_size",
            None,
        )
        if isinstance(text_config, int):
            return text_config

        raise ValueError(
            "Could not infer observation_feature_dim from the base pi0.5 model. "
            "Set policy.observation_feature_dim explicitly."
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """只保存 DSRL head 和配置，不重复保存冻结的 base pi0.5 权重。

        base pi0.5 权重大约数 GB，重复保存会浪费磁盘，也会让 checkpoint 难以管理。
        checkpoint 只记录 `base_policy_path`，加载时再从原路径恢复 base。
        """
        self.config._save_pretrained(save_directory)
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("base_policy.")
        }
        save_file(state_dict, str(save_directory / SAFETENSORS_SINGLE_FILE))

    def get_optim_params(self) -> dict:
        """为 LeRobot 通用 optimizer 路径返回 actor 参数。

        standalone IQL 脚本会分别优化 actor/critic/value；这个方法主要保持
        `PreTrainedPolicy` 接口兼容。
        """
        return self.actor.parameters()

    def reset(self):
        """清空 `select_action` 使用的动作队列。

        pi0.5 一次输出 action chunk，但外部环境通常一步一步消费动作；队列用于缓存
        chunk 中尚未执行的 action。
        """
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def set_return_to_go_stats(self, mean: Tensor | float, std: Tensor | float) -> None:
        """保存从离线 cache metadata 加载的 return 归一化统计量。"""
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.return_to_go_mean.device).reshape(1)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=self.return_to_go_std.device).reshape(1)
        self.return_to_go_mean.copy_(mean_t)
        self.return_to_go_std.copy_(std_t.clamp_min(1e-6))

    def _reshape_noise(self, noise: Tensor) -> Tensor:
        """把 latent noise 统一整理成 `[B, chunk_size, max_action_dim]`。

        cache 为了节省存储和方便 MLP 训练会保存 flattened noise；解码动作时则需要
        chunk 形状。这个函数集中处理两种表示，避免各处重复 reshape。
        """
        if noise.ndim == 2:
            return noise.view(noise.shape[0], self.config.chunk_size, self.config.max_action_dim)
        if noise.ndim == 3:
            expected_shape = (self.config.chunk_size, self.config.max_action_dim)
            if noise.shape[1:] != expected_shape:
                raise ValueError(f"Expected noise shape (*, {expected_shape}), got {tuple(noise.shape)}")
            return noise
        raise ValueError(f"Expected noise tensor with rank 2 or 3, got shape {tuple(noise.shape)}")

    def _project_noise(self, noise: Tensor) -> Tensor:
        """把 noise 保持在 pi0.5 原始高斯 latent space，并可选做宽松裁剪。

        pi0.5 推理时初始 noise 来自标准高斯 `sample_noise()`，不是 tanh-squashed 动作。
        因此这里不做 tanh，只在配置了 `noise_action_magnitude` 时做 clamp，防止优化或 actor
        输出极端离群值。
        """
        if self.config.noise_action_magnitude is None:
            return noise
        return noise.clamp(-self.config.noise_action_magnitude, self.config.noise_action_magnitude)

    def _sample_pi05_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        """复用 pi0.5 的标准高斯 noise 初始化函数。

        这样 action-to-noise 反演的初始点和 pi0.5 原始推理阶段完全一致：
        都是 `[B, chunk_size, max_action_dim]` 形状的 N(0, 1) latent noise。
        """
        return self.base_policy.model.sample_noise(shape, device)

    def flatten_noise(self, noise: Tensor) -> Tensor:
        """把 chunk 形状的 noise 展平成 critic 可以拼接的向量。"""
        return self._reshape_noise(noise).reshape(noise.shape[0], -1)

    def _prepare_base_inputs(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
        """从已经 processor 处理过的 LeRobot batch 中取出 pi0.5 prefix 输入。"""
        images, img_masks = self.base_policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        return images, img_masks, tokens, masks

    @torch.no_grad()
    def _encode_observation_context(self, batch: dict[str, Tensor]) -> Tensor:
        """把观测编码成 actor、critic、value head 共用的定长特征。

        这里只跑 pi0.5 的 prefix path（图像 + 语言 + state），不跑 action suffix。
        返回值是对有效 prefix token hidden states 做 mask mean pooling 后的 float32 表征。
        """
        images, img_masks, tokens, masks = self._prepare_base_inputs(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.base_policy.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.base_policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        # 真实 bfloat16 pi0.5 在 SDPA 上会遇到 attention bias dtype 限制；eager 路径更稳。
        if hasattr(self.base_policy.model.paligemma_with_expert.paligemma.model.language_model.config, "_attn_implementation"):
            self.base_policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_hidden, _), _ = self.base_policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        prefix_hidden = prefix_hidden.to(dtype=torch.float32)
        mask = prefix_pad_masks.to(dtype=prefix_hidden.dtype).unsqueeze(-1)
        # mask mean pooling 去掉 padding token 的影响，让不同文本长度/图像 token 数下表征尺度稳定。
        return (prefix_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @torch.no_grad()
    def _build_prefix_cache(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, object]:
        """为重复 denoising 或 action-to-noise 反演构建 prefix KV cache。

        解码一个 noise 需要多步 denoising；prefix observation 不变，所以先缓存 KV，
        后续每个 denoising step 只跑 action suffix，节省大量计算。
        """
        images, img_masks, tokens, masks = self._prepare_base_inputs(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.base_policy.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.base_policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        if hasattr(self.base_policy.model.paligemma_with_expert.paligemma.model.language_model.config, "_attn_implementation"):
            self.base_policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.base_policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_pad_masks, past_key_values

    def _decode_noise_from_prefix_cache(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: object,
        noise: Tensor,
        num_steps: int | None = None,
    ) -> Tensor:
        """用冻结的 pi0.5 把 latent noise 解码成可执行的 action chunk。

        pi0.5 的 flow matching 推理从 noise 出发，沿时间从 1 积分到 0。
        actor 学到的 z 就作为这个初始 noise，最终 x_t 的前 `original_action_dim`
        维是真实机器人动作。
        """
        base_model = self.base_policy.model
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        # x_t 是当前 flow 状态，初始化为 actor 或 inversion 给出的 latent noise。
        x_t = self._reshape_noise(noise)
        batch_size = x_t.shape[0]
        device = x_t.device
        # 从 t=1 到 t=0 做 Euler 积分，所以 dt 是负数。
        dt = -1.0 / num_steps

        if hasattr(base_model.paligemma_with_expert.gemma_expert.model.config, "_attn_implementation"):
            base_model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        for step in range(num_steps):
            time = 1.0 + step * dt
            timestep = torch.full((batch_size,), time, dtype=torch.float32, device=device)
            # denoise_step 预测 velocity v_t，然后用 Euler update 推进 x_t。
            v_t = base_model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + dt * v_t

        original_action_dim = self.config.output_features[ACTION].shape[0]
        # pi0.5 内部 action padding 到 max_action_dim；环境只需要原始 action 维度。
        return x_t[:, :, :original_action_dim]

    def decode_noise_to_actions(self, batch: dict[str, Tensor], noise: Tensor, num_steps: int | None = None) -> Tensor:
        """便捷封装：先构建 prefix cache，再把 noise 解码成 action chunk。"""
        prefix_pad_masks, past_key_values = self._build_prefix_cache(batch)
        return self._decode_noise_from_prefix_cache(prefix_pad_masks, past_key_values, noise, num_steps=num_steps)

    def predict_noise(self, batch: dict[str, Tensor]) -> Tensor:
        """从已处理 batch 出发运行 actor，输出 chunk 形状的 latent noise。"""
        obs_features = self._encode_observation_context(batch)
        return self.predict_noise_from_features(obs_features)

    def predict_noise_from_features(self, obs_features: Tensor) -> Tensor:
        """从缓存好的观测特征出发运行 actor，输出 latent noise。"""
        # actor 直接输出 pi0.5 原始 noise space 里的 latent action；
        # 只做可选 clamp，不做 tanh squash，保持和 pi0.5 sample_noise 语义一致。
        actor_output = self.actor(obs_features)
        return self._project_noise(actor_output).view(-1, self.config.chunk_size, self.config.max_action_dim)

    def critic_forward(self, obs_features: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
        """计算一个 state-latent-action 对应的 twin Q 值。"""
        # critic 输入必须同时看到状态表征和 latent action，否则无法判断这个 noise 在当前状态下好不好。
        flat_noise = self.flatten_noise(noise).to(device=obs_features.device, dtype=obs_features.dtype)
        critic_input = torch.cat([obs_features, flat_noise], dim=-1)
        q1 = self.critic1(critic_input).squeeze(-1)
        q2 = self.critic2(critic_input).squeeze(-1)
        return q1, q2

    def value_forward(self, obs_features: Tensor) -> Tensor:
        """计算 IQL value regression 使用的 V(s)。"""
        return self.value_head(obs_features).squeeze(-1)

    def compute_iql_actor_weights(self, advantage: Tensor) -> Tensor:
        """把 IQL advantage 转成 actor 回归 demo latent action 的样本权重。"""
        # advantage 越高，说明该 demo latent action 比当前 V(s) 更好，actor 应该更用力拟合。
        return torch.exp(self.config.iql_temperature * advantage).clamp(max=self.config.iql_adv_clip)

    def compute_recon_quality_weights(self, recon_error: Tensor) -> Tensor:
        """把 action-to-noise 的重建误差转换成可选的样本质量权重。

        valid/invalid 是硬过滤：超过 threshold 的样本直接丢弃。
        quality weight 是软加权：已经 valid 的样本中，recon_error 越小，说明 noise_label
        越能被 frozen pi0.5 解码回 demo action，因此 actor 拟合时可以给更高权重。
        """
        recon_error = recon_error.reshape(-1).to(dtype=torch.float32)
        # exp(-error / temperature) 的范围在 (0, 1]，error=0 时权重为 1。
        # clamp_min 避免极小权重导致 batch 权重和接近 0，引起 loss 归一化不稳定。
        return torch.exp(-recon_error / self.config.recon_error_weight_temperature).clamp_min(1e-6)

    def compute_actor_loss_from_features(
        self,
        obs_features: Tensor,
        target_noise: Tensor,
        sample_weight: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """计算 actor 对 latent-noise 标签的加权回归 loss。

        IQL 和 AWR 最终都会落到“更重地拟合高价值 demo noise”这个目标上；
        区别只在 sample_weight 来自 advantage 还是 return-to-go。
        """
        # 用 actor 当前输出的 latent noise 对齐 action-to-noise 反演得到的 target noise。
        predicted_noise = self.predict_noise_from_features(obs_features)
        target_noise = self._reshape_noise(target_noise).to(device=predicted_noise.device, dtype=predicted_noise.dtype)
        # 每个样本单独求 MSE，后面才能乘 IQL/AWR 的样本权重。
        per_sample_loss = F.mse_loss(predicted_noise, target_noise, reduction="none").mean(dim=(1, 2))

        if sample_weight is None:
            # 没有 policy improvement 权重时退化成普通 latent behavior cloning。
            sample_weight = torch.ones_like(per_sample_loss)
        else:
            # 保证权重和 loss 在同一 device/dtype，避免 CUDA 或混合精度下报错。
            sample_weight = sample_weight.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype).reshape(-1)

        # 用权重和归一化，避免 batch 中权重整体变大时 loss scale 不稳定。
        loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        loss_dict = {
            "loss": loss.item(),
            "mse_loss": per_sample_loss.mean().item(),
            "mean_weight": sample_weight.mean().item(),
        }
        return loss, loss_dict

    def compute_critic_loss(
        self,
        obs_features: Tensor,
        behavior_noise: Tensor,
        reward: Tensor,
        next_obs_features: Tensor,
        done: Tensor,
    ) -> tuple[Tensor, dict]:
        """计算 twin critic 的 Bellman regression loss。

        离线 DSRL 的强化学习信号体现在这里：Q(s,z) 拟合 sparse reward 加
        discount * V(s')，而不是只做 supervised imitation。
        """
        # reward/done 来自 dsrl_pi0-style sparse reward table，需要拉平成 `[B]`。
        reward = reward.reshape(-1).to(device=obs_features.device, dtype=obs_features.dtype)
        done = done.reshape(-1).to(device=obs_features.device, dtype=obs_features.dtype)
        behavior_noise = self._reshape_noise(behavior_noise).to(device=obs_features.device, dtype=obs_features.dtype)

        # critic 评估数据集中真实 demo latent action 的 Q 值。
        q1, q2 = self.critic_forward(obs_features, behavior_noise)
        with torch.no_grad():
            # IQL 使用 V(s') 做 bootstrap target，避免在离线数据外做 action maximization。
            target_v = self.value_forward(next_obs_features)
            td_target = reward + (1.0 - done) * self.config.discount * target_v

        # 两个 critic 都拟合同一个 TD target，这是 twin-Q 版 IQL 的常见写法。
        # 注意这里用“平均”而不是“直接求和”：两者优化目标方向一致，但求和会把
        # critic 梯度规模放大 2 倍，等价于隐式提高 critic learning rate，不利于调参。
        critic1_loss = F.mse_loss(q1, td_target)
        critic2_loss = F.mse_loss(q2, td_target)
        critic_loss = 0.5 * (critic1_loss + critic2_loss)
        loss_dict = {
            "critic_loss": critic_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "target_q_mean": td_target.mean().item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }
        return critic_loss, loss_dict

    def compute_value_loss(
        self,
        obs_features: Tensor,
        behavior_noise: Tensor,
    ) -> tuple[Tensor, dict]:
        """计算 IQL value head 的 expectile regression loss。

        V(s) 不直接取 max Q，而是用 expectile 拟合 dataset action 的 Q，
        这样可以在纯离线数据上做 policy improvement，降低 out-of-distribution 风险。
        """
        behavior_noise = self._reshape_noise(behavior_noise).to(device=obs_features.device, dtype=obs_features.dtype)
        with torch.no_grad():
            # 对 demo latent action 取 conservative 的 min(Q1,Q2)，作为 value regression 的目标。
            q1, q2 = self.critic_forward(obs_features, behavior_noise)
            target_q = torch.minimum(q1, q2)

        value = self.value_forward(obs_features)
        # diff=Q-V；expectile>0.5 会让 V 更贴近高 Q 样本，为 actor 提供 advantage 信号。
        diff = target_q - value
        value_loss = _expectile_loss(diff, self.config.iql_expectile).mean()
        loss_dict = {
            "value_loss": value_loss.item(),
            "value_mean": value.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }
        return value_loss, loss_dict

    def compute_iql_actor_loss(
        self,
        obs_features: Tensor,
        behavior_noise: Tensor,
        sample_weight: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """计算 IQL actor loss，即 advantage-weighted latent behavior cloning。

        actor 不直接最大化 critic，而是按 exp(advantage) 加权拟合数据中的 latent noise；
        这是 IQL 在离线数据上避免 OOD action 的关键。
        """
        behavior_noise = self._reshape_noise(behavior_noise).to(device=obs_features.device, dtype=obs_features.dtype)
        with torch.no_grad():
            # advantage 只作为权重，不对 critic/value 反传，避免 actor step 改动 Q/V 目标。
            q1, q2 = self.critic_forward(obs_features, behavior_noise)
            value = self.value_forward(obs_features)
            advantage = torch.minimum(q1, q2) - value
            actor_weight = self.compute_iql_actor_weights(advantage)
            if sample_weight is not None:
                # 可选 recon_error 质量权重只改变 actor 对 noise_label 的拟合强度，
                # 不改变 critic/value 的 Bellman 目标。
                actor_weight = actor_weight * sample_weight.to(
                    device=actor_weight.device, dtype=actor_weight.dtype
                ).reshape(-1)

        loss, loss_dict = self.compute_actor_loss_from_features(
            obs_features=obs_features,
            target_noise=behavior_noise,
            sample_weight=actor_weight,
        )
        loss_dict["adv_mean"] = advantage.mean().item()
        loss_dict["adv_max"] = advantage.max().item()
        return loss, loss_dict

    def compute_awr_weights(self, return_to_go: Tensor) -> Tensor:
        """用缓存中的 return-to-go 计算 AWR fallback 权重。"""
        # AWR 不训练 critic/value 时可用；这里保留为简单 baseline 和消融实验入口。
        normalized_return = (return_to_go.reshape(-1) - self.return_to_go_mean) / self.return_to_go_std.clamp_min(1e-6)
        return torch.exp(self.config.awr_beta * normalized_return).clamp(min=1.0, max=self.config.awr_weight_clip)

    def invert_actions_to_noise(
        self,
        batch: dict[str, Tensor],
        target_actions: Tensor,
        num_steps: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """把示范动作反演成 pi0.5 flow 初始 latent noise 标签。

        离线数据只有 action，没有 latent noise；DSRL 需要在 latent space 训练 actor/critic，
        所以先固定 pi0.5 decoder，通过优化 noise 让解码后的动作重建 demo action。
        换句话说，这里求解的是下面这个优化问题：

            noise_label = argmin_z MSE(pi0.5_decode(observation, z), demo_action)

        这样做的原因是：dsrl_pi0 的在线版本在 rollout 时本来就知道 SAC 采样出来的 noise，
        replay buffer 可以直接存 noise；但你的离线折衣服数据只保存了机器人真实 action，
        没保存当时的 pi0.5 noise。因此第一阶段必须先把 action 反投影回 pi0.5 的 latent space，
        后续 IQL/AWR 才能把这个 noise 当作 RL action 来训练 Q(s, z)、V(s) 和 actor(s)->z。
        """
        # target_actions 是数据集中真实执行/示范的 action chunk。
        # 反演过程中只把它当监督目标，不对 action 本身做优化。
        target_actions = target_actions.to(dtype=torch.float32)
        # 反演阶段每个 Adam step 都要调用一次 pi0.5 decoder。
        # 如果不单独指定 num_steps，就使用配置里的 latent_inversion_decode_steps；
        # 它通常比正式推理的 num_inference_steps 小很多，用来把 cache 生成时间控制在可接受范围。
        if num_steps is None:
            num_steps = self.config.latent_inversion_decode_steps
        # 对同一个 observation，会在多个优化 step、多个 restart 中反复尝试不同 noise。
        # observation 的图像/语言/state prefix 不变，因此先构建 prefix KV cache。
        # 这样每次只需要跑 action suffix 的 denoising，避免重复编码视觉语言上下文。
        prefix_pad_masks, past_key_values = self._build_prefix_cache(batch)
        batch_size = target_actions.shape[0]
        device = target_actions.device

        # best_noise 保存当前 batch 中每个样本目前找到的最优 noise label。
        # 初始化为 0 只是占位；真正返回前会被各个 restart 中误差更小的 candidate 覆盖。
        best_noise = torch.zeros(
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
            device=device,
            dtype=torch.float32,
        )
        # best_error 记录每个样本最小的重建误差。
        # 用 inf 初始化，保证第一个 restart 的结果一定会被接受。
        best_error = torch.full((batch_size,), torch.inf, device=device, dtype=torch.float32)

        for restart_idx in range(self.config.latent_restarts):
            # flow/diffusion decoder 从 noise 到 action 的映射不是线性的，也不保证是单峰优化问题。
            # 同一个 demo action 附近可能存在多个可行 noise，随机初始化可能落到不同局部最优。
            # 因此做多次 restart，最后按重建误差选择最好的 noise label。
            # 初始化必须复用 pi0.5 的 sample_noise，使反演起点和原始 pi0.5 推理一致。
            initial_noise = self._sample_pi05_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                device,
            )
            latent = nn.Parameter(
                initial_noise.to(device=device, dtype=torch.float32)
            )
            # 这里只优化 latent 这个“输入变量”，不优化 pi0.5 模型参数。
            # 目的不是让 base policy 学 demo，而是为每条 demo 找一个能被当前 base policy 解码的 noise 标签。
            optimizer = torch.optim.Adam([latent], lr=self.config.latent_inversion_lr)

            # patience 早停只控制当前 restart 的优化长度，不改变 max_steps 上限。
            # best_restart_error 记录 batch 平均重建误差的历史最好值；
            # steps_without_improvement 记录连续多少步没有超过 min_delta 的有效改善。
            best_restart_error = torch.tensor(torch.inf, device=device)
            steps_without_improvement = 0
            for _step in range(self.config.latent_inversion_steps):
                optimizer.zero_grad(set_to_none=True)
                # latent 就是要优化的 pi0.5 初始 noise，本身来自标准高斯初始化。
                # 这里不再 tanh squash，否则 noise label 会偏离 pi0.5 原始 noise 分布；
                # 只做可选 clamp，防止优化过程中出现极端离群值。
                # noise 的形状是 [batch_size, chunk_size, max_action_dim]，对应一个完整 action chunk 的 latent。
                noise = self._project_noise(latent)
                # fixed observation + current noise -> predicted action chunk。
                # 这里调用的是冻结的 pi0.5 flow decoder，因此梯度会从 action 重建误差传回 noise/latent，
                # 但不会更新 pi0.5 的权重。
                predicted_actions = self._decode_noise_from_prefix_cache(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    noise=noise,
                    num_steps=num_steps,
                )
                # 重建误差衡量“当前 noise 能不能解释这条 demo action”。
                # reduction="none" 后再按 chunk/action 维度求均值，是为了保留 batch 内每个样本自己的误差。
                # 这些 per-sample error 后面会用于过滤无效 cache 样本。
                recon_error = F.mse_loss(predicted_actions, target_actions, reduction="none").mean(dim=(1, 2))
                # L2 正则避免 optimizer 通过极端 noise 勉强拟合 action。
                # 如果没有这个约束，反演可能得到 pi0.5 正常采样分布之外的 z，
                # 后续 actor 学这种 z 会更难泛化，critic 的 Q(s,z) 也更容易不稳定。
                regularizer = noise.square().mean(dim=(1, 2))
                # batch 内取均值后反传，这样一次 backward 可以同时优化 batch 中所有样本的 latent。
                loss = (recon_error + self.config.latent_reg_weight * regularizer).mean()
                loss.backward()
                optimizer.step()

                if self.config.latent_inversion_log_freq > 0 and (
                    _step == 0
                    or (_step + 1) % self.config.latent_inversion_log_freq == 0
                    or _step == self.config.latent_inversion_steps - 1
                ):
                    # 低频输出反演内层优化状态，用来判断 recon_error 是否在下降。
                    # 这里只打印 batch 统计，不打印每个样本，避免日志过大。
                    _safe_tqdm_write(
                        "[latent inversion] "
                        f"restart={restart_idx + 1}/{self.config.latent_restarts} "
                        f"step={_step + 1}/{self.config.latent_inversion_steps} "
                        f"loss={loss.detach().item():.6f} "
                        f"recon_mean={recon_error.detach().mean().item():.6f} "
                        f"recon_min={recon_error.detach().min().item():.6f} "
                        f"recon_max={recon_error.detach().max().item():.6f} "
                        f"reg={regularizer.detach().mean().item():.6f}"
                    )

                # 早停判断使用 batch 平均 recon_error，而不是带正则的 loss。
                # 原因是 noise_label 的有效性最终由 recon_error < threshold 决定，
                # 所以停止条件应该直接观察动作重建质量是否还在明显改善。
                current_error = recon_error.detach().mean()
                improvement = best_restart_error - current_error
                if improvement > self.config.latent_inversion_min_delta:
                    best_restart_error = current_error
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                if (
                    self.config.latent_inversion_patience is not None
                    and steps_without_improvement >= self.config.latent_inversion_patience
                ):
                    # 最近 patience 步都没有明显改善，继续跑到 max_steps 大概率只会浪费时间。
                    if self.config.latent_inversion_log_freq > 0:
                        _safe_tqdm_write(
                            "[latent inversion] "
                            f"restart={restart_idx + 1}/{self.config.latent_restarts} "
                            f"early_stop_step={_step + 1} "
                            f"best_recon_mean={best_restart_error.detach().item():.6f}"
                        )
                    break

            with torch.no_grad():
                # 一个 restart 结束后，再用最终 latent 评估一次 candidate。
                # 这里 no_grad 是因为已经不需要继续优化，只做选择和保存。
                candidate_noise = self._project_noise(latent)
                predicted_actions = self._decode_noise_from_prefix_cache(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    noise=candidate_noise,
                    num_steps=num_steps,
                )
                candidate_error = F.mse_loss(predicted_actions, target_actions, reduction="none").mean(dim=(1, 2))
                # better 是按样本比较，而不是要求整个 batch 同时变好。
                # 这样 batch 中第 0 个样本可以选择 restart A，第 1 个样本可以选择 restart B。
                better = candidate_error < best_error
                best_error = torch.where(better, candidate_error, best_error)
                # better[:, None, None] 把 `[B]` mask 扩展到 `[B, chunk_size, max_action_dim]`，
                # 对每个样本独立替换完整 noise chunk。
                best_noise = torch.where(better[:, None, None], candidate_noise.detach(), best_noise)

        # 返回的 best_noise 会写入 cache 作为 noise_label；
        # best_error 会写入 recon_error，用 latent_recon_threshold 判断这个 label 是否可信。
        return best_noise, best_error

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """兼容 LeRobot 普通训练接口的 actor-only 前向。

        独立离线 DSRL 脚本会显式调用 critic/value/IQL loss；这个 forward 主要用于测试、
        简单 AWR/BC 路径，以及未来接入通用 trainer 时仍能返回标准 `(loss, log_dict)`。
        """
        # 先用冻结 pi0.5 编码观测，确保 actor 训练和推理使用同一套状态表征。
        obs_features = self._encode_observation_context(batch)
        # `noise_label` 是 precompute 阶段 action-to-noise 反演得到的监督目标。
        target_noise = batch["noise_label"]
        if "sample_weight" in batch:
            # 外部已经计算好权重时直接使用，方便测试或自定义 offline RL 算法。
            sample_weight = batch["sample_weight"]
        elif "return_to_go" in batch:
            # 没有显式权重但带 return 时，退化成 AWR 权重。
            sample_weight = self.compute_awr_weights(batch["return_to_go"])
        else:
            # 没有 RL 权重时就是普通 latent BC。
            sample_weight = None
        if self.config.recon_error_weighting and "recon_error" in batch:
            # 可选质量加权：valid 样本仍然全部可训练，但重建误差更小的 noise_label 权重更高。
            recon_weight = self.compute_recon_quality_weights(batch["recon_error"]).to(target_noise.device)
            sample_weight = recon_weight if sample_weight is None else sample_weight.to(recon_weight.device) * recon_weight

        if reduction == "none":
            # 有些通用评估代码需要逐样本 loss，因此保留不聚合分支。
            predicted_noise = self.predict_noise_from_features(obs_features)
            target_noise = self._reshape_noise(target_noise).to(device=predicted_noise.device, dtype=predicted_noise.dtype)
            per_sample_loss = F.mse_loss(predicted_noise, target_noise, reduction="none").mean(dim=(1, 2))
            return per_sample_loss, {"loss": per_sample_loss.mean().item(), "mse_loss": per_sample_loss.mean().item()}

        return self.compute_actor_loss_from_features(
            obs_features=obs_features,
            target_noise=target_noise,
            sample_weight=sample_weight,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """预测一个完整 action chunk。

        推理时 actor 先选 latent noise，冻结 pi0.5 再把 noise 解码为真实机器人动作；
        也允许通过 kwargs 传入指定 noise，便于调试反演质量或做消融。
        """
        noise = kwargs.get("noise")
        if noise is None:
            # 常规路径：actor 根据当前观测自主生成 latent noise。
            noise = self.predict_noise(batch)
        return self.decode_noise_to_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """按环境一步一步返回动作。

        pi0.5 一次生成 chunk，但机器人控制循环通常每次只要一个 action；
        队列为空时重新预测 chunk，之后每次弹出下一步动作。
        """
        if len(self._action_queue) == 0:
            # 只缓存 n_action_steps 个动作，和 pi0.5/LeRobot 的 chunk 执行协议一致。
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
