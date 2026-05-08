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

import builtins
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    """
    Extra keyword arguments accepted by chunked inference.
    chunk 推理阶段允许透传的额外参数。
    """

    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    # 针对不同设备做 dtype 兜底，避免请求了后端不支持的精度。
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        # CPU 对 bfloat16 的支持不稳定，因此这里显式退回 float32。
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (generalized for per-token time)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Accepts `time` of shape `[B]` (global time, original behavior) or `[B, T]` (per-token time,
    used by Training-Time RTC). Output shape is `[B, dim]` or `[B, T, dim]` respectively.
    
    这个函数既支持原始 PI0.5 的“每个 batch 一个时间”，
    也支持 training-time RTC 的“每个 token 一个时间”。
    为什么需要 per-token time：
    training-time RTC 会让同一个 action chunk 里同时存在两类 token：
    - frozen prefix 已经是 clean action；
    - suffix 仍然处在某个 flow-matching 噪声时间 t。
    如果整个 chunk 只能共享一个 t，模型就无法知道“前缀已经干净、后缀还要去噪”。
    因此这里允许 time=[B,T]，让每个动作 token 带自己的噪声/干净状态。
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim not in (1, 2):
        raise ValueError(
            f"The time tensor is expected to be of shape `(batch_size,)` or `(batch_size, seq_len)`, "
            f"got shape {tuple(time.shape)}"
        )

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Broadcast scaling_factor: [half_dim] with time: [B] or [B, T]
    # 把频率尺度广播到全 batch / 全 token 上，得到对应时间位置的正余弦输入。
    scaling_factor = 1.0 / period * 2 * math.pi  # [half_dim]
    # time.unsqueeze(-1): [B, 1] or [B, T, 1]; result: [B, half_dim] or [B, T, half_dim]
    sin_input = time.unsqueeze(-1) * scaling_factor
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    # MPS 后端在底层 Beta/Dirichlet 采样上有限制，因此这里先在 CPU 采样再搬回目标设备。
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    
    pad_masks 决定 token 是否真实存在，att_masks 决定它在注意力图里属于哪种可见性分组。
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    
    把状态或动作补齐到模型内部固定维度，便于统一投影层处理。
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    
    这个函数会先按比例缩放，再补黑边，因此不会拉伸图像内容。
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    """
    Compute one full transformer layer jointly for prefix and suffix streams.
    联合计算一层 transformer，在 prefix 和 suffix 两条流上同时前向。

    Why this exists:
        In the joint path, pi05 does not simply run the VLM stack and the action
        expert completely separately. Instead, at each layer it forms one shared
        attention computation over the concatenated prefix/suffix tokens, then
        splits the result back to each branch.
        在联合路径里，pi05 不是把 prefix 和 suffix 完全各跑各的。
        它会在每一层先把两边的 token 拼起来做一次共享注意力，再把结果拆回两条分支。

    Why it is a standalone function:
        It is defined outside the class so it can be passed cleanly into
        ``torch.utils.checkpoint.checkpoint(...)``.
        之所以单独写成函数，是为了方便直接传给
        ``torch.utils.checkpoint.checkpoint(...)`` 做 gradient checkpoint。
    """
    models = [paligemma.model.language_model, gemma_expert.model]
    # inputs_embeds[0]: prefix hidden states, shape [B, N_prefix, D]
    # inputs_embeds[1]: suffix hidden states, shape [B, N_suffix, D]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        # 对 prefix 分支和 suffix 分支分别做本层的 layernorm + qkv 投影。
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        # hidden_states: [B, N_i, D]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        # q/k/v after reshape+transpose: [B, num_heads, N_i, head_dim]
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    # 这里是联合建模的关键：
    # 把 prefix/suffix 两边的 q/k/v 在 token 维拼起来，做一次共享注意力。
    # concatenated q/k/v: [B, num_heads, N_prefix + N_suffix, head_dim]
    query_states = torch.cat(query_states, dim=2) # torch.Size([2, 8, 1018, 256])
    key_states = torch.cat(key_states, dim=2)   # torch.Size([2, 1, 1018, 256])
    value_states = torch.cat(value_states, dim=2) # torch.Size([2, 1, 1018, 256])
    dummy_tensor = torch.zeros(      # torch.Size([2, 1018, 256])
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    # 在共享注意力图上统一算注意力，prefix 和 suffix 都能通过这一步交换信息。
    # attention_mask: [B, 1, Q, K], typically Q=K=N_prefix+N_suffix
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    # att_output after merge heads: [B, N_prefix + N_suffix, D]
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        # 再按各自原来的 token 长度把共享注意力输出切回 prefix/suffix 两段。
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        # out_emb: [B, N_i, D], where i is prefix or suffix branch
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""
    # 这里是一个轻量配置壳，用来把当前文件需要的 Gemma 结构参数组织起来。

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    # 根据字符串别名切换到不同规模的 Gemma 配置。
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""
    # 这是 pi05 的双路径主体：
    # prefix 路径处理图像+语言，suffix 路径处理动作 expert。

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        # use_adarms=[vlm_use_adarms, expert_use_adarms]
        # pi05 里通常只给动作 expert 路径打开 AdaRMS。
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        # Start from the HuggingFace PaliGemma config shell, then overwrite the parts
        # needed by OpenPI / PI0.5.
        # 中文说明：
        # 这里先拿到一个 HuggingFace 默认的 PaliGemma 配置壳，然后把 PI0.5
        # 真正需要的文本塔、视觉塔、多模态投影参数逐项覆盖进去。
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()

        # `_vocab_size` is an internal HF field for total tokenizer vocabulary size.
        # 中文：整个 tokenizer 的词表大小。这里和 image token 一起对齐到 PaliGemma 预训练设定。
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        # `image_token_index` is the special placeholder token id reserved for image slots.
        # 中文：图像占位 token 的 id。多模态输入里，语言模型会用这个特殊 token 位置来对齐视觉特征。
        vlm_config_hf.image_token_index = 257152

        # Text decoder architecture parameters.
        # 中文：下面这组字段决定 prefix 里的语言 decoder 结构规模。
        # `hidden_size`: token hidden state width D, i.e. embedding/decoder main channel width.
        # 中文：主隐藏维度 D，几乎所有 `[B, T, D]` 里的 D 都来自它。
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        # `intermediate_size`: MLP expansion width inside each transformer block.
        # 中文：前馈网络中间层维度，通常比 hidden_size 更大。
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        # `num_attention_heads`: number of query heads in self-attention.
        # 中文：注意力头数，决定 Q 会被分成多少个 head。
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        # `head_dim`: width of each attention head.
        # 中文：每个注意力头的维度；通常 `hidden_size = num_attention_heads * head_dim`
        # 或至少在量级上由这两者共同决定。
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        # `num_hidden_layers`: decoder depth.
        # 中文：transformer 层数，也就是 prefix 路径 decoder 有多少层。
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        # `num_key_value_heads`: number of KV heads for grouped-query attention / MQA.
        # 中文：K/V 头数，可能小于 query 头数，用于 GQA/MQA 以节省显存和计算。
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        # `hidden_activation`: activation function used inside the MLP.
        # 中文：前馈网络里的激活函数类型。
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        # `dtype`: initialization / expected computation dtype recorded in config.
        # 中文：配置层记录的默认精度声明；真正参数后面还会根据 `precision` 再转换。
        vlm_config_hf.text_config.dtype = "float32"
        # `vocab_size`: text embedding / LM head vocabulary size.
        # 中文：文本 embedding 表和语言建模头对应的词表大小。
        vlm_config_hf.text_config.vocab_size = 257152
        # `use_adarms`: whether the text-side decoder should use adaptive RMSNorm.
        # 中文：prefix 侧语言 decoder 是否启用 AdaRMS。
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        # `adarms_cond_dim`: width of the conditioning vector used by AdaRMS.
        # 中文：如果启用 AdaRMS，cond 向量的维度是多少；这里通常与 hidden_size 对齐。
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None

        # Vision tower / projector parameters.
        # 中文：下面这组字段决定视觉编码器和视觉到语言空间的投影方式。
        # `image_size`: expected square image resolution for the vision tower.
        # 中文：视觉塔期望输入的图像边长。
        vlm_config_hf.vision_config.image_size = image_size
        # `intermediate_size`: MLP width inside the vision transformer blocks.
        # 中文：视觉塔内部前馈层宽度，不是最终输出 token 维度。
        vlm_config_hf.vision_config.intermediate_size = 4304
        # `projection_dim`: width after the multimodal projector, i.e. visual token dim before/at fusion.
        # 中文：视觉特征经过多模态投影后的通道数，用来把视觉特征接到语言空间。
        vlm_config_hf.vision_config.projection_dim = 2048
        # `projector_hidden_act`: activation used by the multimodal projector.
        # 中文：视觉 projector 内部使用的激活函数。
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        # `dtype`: recorded precision hint for the vision side.
        # 中文：视觉配置记录的默认精度；后面依然会按实际 precision 策略转换。
        vlm_config_hf.vision_config.dtype = "float32"

        # Build the action expert config from a plain Gemma config template.
        # 中文说明：
        # suffix / action expert 不是完整的 PaliGemma，而是一套 Gemma decoder。
        # 它专门负责处理动作 token，因此这里直接从 GemmaConfig 构造。
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            # Same meaning as the text_config fields above, but now for the action expert branch.
            # 中文：下面这些字段和上面 text_config 同义，只是现在作用在动作 expert 分支。
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            # Keep tokenizer-space size aligned with PaliGemma even though the action expert
            # later consumes embeddings rather than normal text token ids.
            # 中文：这里仍保留词表大小字段，保持和 Gemma/PaliGemma 配置接口兼容。
            vocab_size=257152,
            # MLP activation in the action expert.
            # 中文：动作 expert 前馈网络的激活函数。
            hidden_activation="gelu_pytorch_tanh",
            # Start from float32 config; later `to_bfloat16_for_selected_params(...)` may cast weights.
            # 中文：配置层先登记为 float32，后面再按精度策略实际转换权重。
            dtype="float32",
            # Whether the action expert uses AdaRMS.
            # 中文：动作 expert 是否启用 AdaRMS；pi05 一般是这里开启。
            use_adarms=use_adarms[1],
            # Conditioning width for AdaRMS on the action expert side.
            # 中文：动作 expert 的 AdaRMS 条件向量维度。
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # `self.paligemma` handles the prefix branch: image + language observation context.
        # 中文：`self.paligemma` 是 prefix 路径，负责图像和语言观测上下文。
        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        # `self.gemma_expert` handles the suffix branch: action tokens / action expert decoding.
        # 中文：`self.gemma_expert` 是 suffix 路径，负责动作 token 的 decoder / expert 计算。
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        # Disable the expert's token embedding table because the action expert does not consume
        # discrete token ids here; upper layers feed continuous suffix embeddings directly.
        # 中文：动作 expert 这里不走“离散 token id -> embedding”这条路，而是上层直接喂入
        # 连续的动作 embedding，所以把 `embed_tokens` 置空，强调这条分支不需要词表嵌入表。
        self.gemma_expert.model.embed_tokens = None

        # Apply the requested runtime precision policy.
        # 中文：先按 `precision` 把大部分参数转到目标精度，例如 bfloat16。
        self.to_bfloat16_for_selected_params(precision)
        # Apply freezing policy such as `freeze_vision_encoder` / `train_expert_only`.
        # 中文：再根据配置决定哪些模块需要冻结，只训练哪一部分。
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """
        Convert most weights to the requested precision while keeping selected
        submodules in float32.
        把大部分权重转到目标精度，但保留部分敏感模块在 float32。
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        """Apply freezing rules to submodules. 对子模块应用冻结策略。"""
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train() to keep frozen modules in eval mode. 重载 train() 以维持冻结模块的 eval 状态。"""
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        # 图像路径强制走 float32，更稳定，最后再转回调用方期望的 dtype。
        # image: [B, C, H, W]
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        # pooler_output / features: [B, N_img, D]
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Embed text token ids into language embeddings. 把文本 token id 映射成语言 embedding。"""
        # tokens: [B, N_text] -> embeddings: [B, N_text, D]
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """
        Run the prefix path, suffix path, or both jointly.
        运行 prefix 路径、suffix 路径，或者两者联合前向。

        This method has three modes:
        1. prefix-only: used to build KV cache from observation context
        2. suffix-only: used during iterative denoising with cached prefix
        3. joint forward: used in training, where prefix and suffix interact layer-by-layer
        这个函数有三种工作模式：
        1. 只跑 prefix：常用于先缓存观测上下文
        2. 只跑 suffix：常用于带 prefix cache 的逐步去噪
        3. 联合前向：训练时最常见，prefix/suffix 在每层共享注意力
        """
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            # 分支 1：只有 prefix，没有 suffix。
            # 典型用途是先把观测上下文编码一遍，并缓存 past_key_values 给后续去噪复用。
            # inputs_embeds[0]: [B, N_prefix, D]
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            # prefix_output: [B, N_prefix, D]
            suffix_output = None
        elif inputs_embeds[0] is None:
            # 分支 2：只有 suffix。
            # 典型用途是“已经有 prefix cache 了”，当前只需要继续更新动作 token。
            # inputs_embeds[1]: [B, N_suffix, D]
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            # suffix_output: [B, N_suffix, D]
            prefix_output = None
            prefix_past_key_values = None
        else:
            # 分支 3：prefix 和 suffix 同时存在。
            # 训练时走这条路径，让两边 token 在每一层共享注意力、充分交换信息。
            # inputs_embeds[0]: [B, N_prefix, D]
            # inputs_embeds[1]: [B, N_suffix, D]
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers  #18

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    # 每一层都可以按需使用 checkpoint，以节省联合前向时的显存。
                    inputs_embeds = torch.utils.checkpoint.checkpoint(   # 一直不变 inputs_embeds len为2的list len[0].shape=torch.Size([2, 968, 2048]) len[1].shape=torch.Size([2, 50, 1024])
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    # 不开 checkpoint 时，就直接执行同一层的联合计算逻辑。
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                # 每层后都保持两条分支的张量形状不变：
                # prefix hidden: [B, N_prefix, D], suffix hidden: [B, N_suffix, D]

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                # 每条分支在所有层跑完后，还要各自过最终 norm，和标准 transformer 一致。
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(  # len 为2，index 0 ：torch.Size([2, 968, 2048])  index 1 ：torch.Size([2, 50, 1024])
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            # prefix_output: [B, N_prefix, D]
            # suffix_output: [B, N_suffix, D]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""
    # 这是底层网络主体；上层 PI05Policy 会再把它包装成 LeRobot 的统一 policy 接口。

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        # prefix 使用 PaliGemma 视觉语言路径，suffix 使用动作 expert 路径。
        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            # 训练时主 forward 也一起 compile。
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """
        Helper method to apply gradient checkpointing if enabled.
        在需要时对一段前向计算应用 gradient checkpointing 的统一辅助函数。

        What it does:
            If gradient checkpointing is enabled and the model is in training mode,
            this wrapper runs ``func(*args, **kwargs)`` through
            ``torch.utils.checkpoint.checkpoint(...)``. Otherwise it calls
            ``func`` directly.
            如果当前开启了 gradient checkpointing，并且模型处于训练模式，
            这个函数就会把 ``func(*args, **kwargs)`` 包进
            ``torch.utils.checkpoint.checkpoint(...)``；否则就直接正常执行 ``func``。

        Why it is needed:
            Several heavy sub-paths in PI0.5, such as image embedding, suffix
            embedding, and transformer forward passes, may optionally use
            checkpointing. Wrapping the switch in one helper avoids repeating
            the same if/else logic everywhere.
            PI0.5 里有多段比较重的前向路径，例如图像 embedding、suffix embedding、
            transformer 主前向，都可能按需打开 checkpoint。把开关统一封装在这里，
            可以避免每个调用点都重复写一遍相同的条件分支。

        How it works:
            Gradient checkpointing trades compute for memory. During the forward
            pass it avoids saving some intermediate activations; during backward,
            PyTorch re-runs the wrapped function to recompute them.
            Gradient checkpointing 的本质是“以计算换显存”。
            前向阶段少保存一部分中间激活；反向传播时，PyTorch 会重新执行一次
            被包装的函数，把这些中间结果再算回来。

        Notes:
            - It is only active when both ``self.gradient_checkpointing_enabled``
              and ``self.training`` are True.
            - ``use_reentrant=False`` selects the newer non-reentrant checkpoint
              implementation.
            - ``preserve_rng_state=False`` avoids saving/restoring RNG state to
              reduce overhead, which is acceptable here because these wrapped
              blocks are intended to be deterministic enough for recomputation.
            - 只有 ``self.gradient_checkpointing_enabled`` 和 ``self.training``
              同时为 True 时才会真的启用。
            - ``use_reentrant=False`` 表示使用新版 non-reentrant 实现。
            - ``preserve_rng_state=False`` 不额外保存/恢复随机数状态，减少开销；
              这里默认假设这些被包装的计算块适合做这种重新计算。
        """
        # 统一封装 checkpoint 开关，避免各个子路径重复写条件判断。
        # This helper centralizes the "use checkpoint or call directly" decision.
        if self.gradient_checkpointing_enabled and self.training:
            # 开启后，不直接保存这段前向的完整中间激活，而是在反向时按需重算。
            # When enabled, activations are recomputed during backward to save memory.
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        # 没开启 checkpoint 时，直接执行原始函数，走正常前向路径。
        # If checkpointing is disabled, just execute the function normally.
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        # HF 注意力常用 [B, 1, Q, K] 形状；无效位置填大负值。
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        """
        Sample Gaussian noise for the internal action representation.
        为模型内部动作表示采样高斯噪声。

        This noise is used both during training and inference:
        - training: to build the interpolated state x_t
        - inference: as the initial latent action chunk before denoising
        这份噪声同时服务于训练和推理：
        - 训练时用来构造中间状态 x_t
        - 推理时作为去噪初始 latent action chunk

        Shape:
            ``shape`` is typically ``[B, T, A_pad]`` where
            - ``B`` = batch size
            - ``T`` = chunk_size
            - ``A_pad`` = max_action_dim
            常见形状是 ``[B, T, A_pad]``，分别表示 batch、大动作块长度、补齐后的动作维度。
        """
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        """Sample flow-matching time values. 采样 flow matching 使用的时间变量。"""
        # 先从 Beta 分布采样，再缩放到配置指定区间。
        # 这样既能控制 t 的分布形状，也能避免直接落在 0/1 这些极端点。
        # 输出 shape: [B]
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        # prefix 对应 observation 侧：图像 + 文本提示。
        # 返回值分别是：
        # - embs: 拼好的 prefix token embeddings
        # - pad_masks: 哪些 prefix token 是有效 token
        # - att_masks: prefix token 的注意力分组编码
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            # img: [B, C, H, W] torch.Size([2, 3, 224, 224])
            # img_mask: [B]
 
            def image_embed_func(img):
                # embed_image() 会把单路相机图像编码成视觉 token。
                # 视觉塔和 projector 都封装在 paligemma_with_expert 里。
                return self.paligemma_with_expert.embed_image(img)

            # 图像编码开销比较大，因此这里也走 _apply_checkpoint()，可以按需省显存。
            img_emb = self._apply_checkpoint(image_embed_func, img) # torch.Size([2, 256, 2048])
            # img_emb: [B, N_img, D]
            bsize, num_img_embs = img_emb.shape[:2]

            # 每个视觉 token 都复制一份这路相机的有效性 mask。
            embs.append(img_emb)
            # expanded image mask: [B, N_img]
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            # prefix 图像 token 都属于“前缀上下文”组，因此 att mask 记 0。
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            # embed_language_tokens() 把 tokenizer 的 token id 变成词向量。
            # 乘 sqrt(hidden_dim) 是 transformer 常见的 embedding scaling。
            # tokens: [B, N_text]
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            # lang_emb before scaling: [B, N_text, D]
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        # 文本 embedding 同样可以按需包进 checkpoint。
        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        # 文本 token 的有效性由 tokenizer 产生的 attention mask 决定。
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        # 文本也属于 prefix 上下文，同样标记为 0。
        att_masks += [0] * num_lang_embs

        # 把多路图像 token 和文本 token 拼成一条统一的 prefix 序列。
        # embs: [B, N_prefix, D]
        embs = torch.cat(embs, dim=1)
        # pad_masks: [B, N_prefix]
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        # 把长度为 N 的模板扩成 [B, N]，供 batch 内每个样本共享使用。
        # att_masks: [B, N_prefix]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        # suffix 对应动作去噪路径：把 noisy action 和时间条件一起映射到 expert token 空间。
        # 这里的 suffix 是“待预测动作块”的 token 化表示。
        # noisy_actions: [B, T, A_pad]
        # timestep: [B] or [B, T]
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        # create_sinusoidal_pos_embedding() 把连续时间 t 编成向量，
        # 让 expert 知道当前是在第几个去噪阶段。
        time_emb = create_sinusoidal_pos_embedding(   # torch.Size([2, 1024])
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        # time_emb: [B, D] or [B, T, D]
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            # action_in_proj 把动作从 action dim 投到 expert hidden dim。
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        # action_emb: [B, T, D]

        def time_mlp_func(time_emb):
            # 这两层 MLP 把“原始时间编码”变成更适合 AdaRMS 条件化使用的向量。
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        # 当前实现没有把时间直接加到 action embedding 上，
        # 而是把它单独作为 expert 的 AdaRMS conditioning。
        action_time_emb = action_emb
        adarms_cond = time_emb
        # adarms_cond: [B, D] or [B, T, D]

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        # suffix 中的动作 token 全部有效，因此这里 mask 全 1。
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        # action_time_mask: [B, T]
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        # 动作 token 使用 suffix 侧的因果结构，不让 prefix token 被动作反向影响。
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        # embs: [B, T, D]
        pad_masks = torch.cat(pad_masks, dim=1)
        # pad_masks: [B, T]
        # 这里保留和 OpenPI 对齐的分组编码格式，后面再交给 make_att_2d_masks 展开。
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        # att_masks after expand: [B, T]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        # 这里返回逐元素 MSE，而不是最终标量 loss。
        # 上层 PI05Policy.forward() 还会继续做 action dim 截断与 RTC mask 归约。
        if noise is None:
            # sample_noise() 的作用：生成和 actions 同 shape 的标准高斯噪声。
            # 为什么需要它：flow matching 训练要把真实动作和噪声插值成中间状态 x_t。
            # 它怎么做到：内部调用 torch.normal(mean=0, std=1, size=shape, device=device)。
            # sample_noise() creates Gaussian noise matching the action tensor shape for flow matching.
            noise = self.sample_noise(actions.shape, actions.device)  # torch.Size([2, 50, 32])

        if time is None:
            # sample_time() 的作用：为每个 batch 样本采一个连续时间 t。
            # 为什么需要它：训练要在不同噪声强度/插值位置上监督模型。
            # 它怎么做到：先按 Beta 分布采样，再缩放到配置指定的时间区间。
            # sample_time() samples the flow-matching time value t for each sample.
            time = self.sample_time(actions.shape[0], actions.device) # torch.Size([8]) 是batch size的大小

        batch_size = actions.shape[0]
        chunk_size = self.config.chunk_size
        device = actions.device

        # ---- Training-Time RTC: per-token time (arXiv 2512.05964) ----
        # 中文说明：
        # training-time RTC 的核心是“冻结前缀 + 每 token 独立时间”，
        # 而不是额外的外部 guidance。
        #
        # 这段代码对应官方 training-time action conditioning 的训练分布：
        # 给模型看的不是“从零生成整个 chunk”，而是“chunk 开头已经有一段确定动作，
        # 你需要接着它往后生成”。这样可以把真实部署时的 train-test gap 前移到训练阶段。
        #
        # 注意 PI05 的时间方向和官方 kinetix/JAX demo 相反：
        # - 官方 demo: time=0 是纯噪声，time=1 是 clean action；
        # - 这里 PI05: x_t = t * noise + (1 - t) * action，所以 time=1 是纯噪声，time=0 是 clean action。
        # 因此 frozen prefix 的 time 必须设成 0，而不是官方 demo 里的 1。
        if self.config.training_rtc:
            K = self.config.simulated_delay  # 设置为5 
            # (a) Sample delay ∈ {0,...,K-1} with exp-decaying weights (smaller delays more likely)
            # delay 表示当前样本里有多少个 token 被当作“上一块 chunk 已经确定的前缀”。
            # 使用 exp 权重不是为了数学必要性，而是为了让训练分布更像真实部署：
            # 小延迟/短前缀更常见，大延迟也会偶尔出现，让模型保持鲁棒。
            w = torch.exp(torch.arange(K, device=device).flip(0).float())
            w = w / w.sum()
            delay = torch.multinomial(w.expand(batch_size, -1), num_samples=1).squeeze(-1)  # [B]

            # (b) Per-token mask: True = frozen prefix
            # rtc_mask[i, j]=True 表示第 i 个样本的第 j 个动作 token 是条件输入，
            # 它不是这一步要学习预测的目标，而是“已知事实”。
            token_indices = torch.arange(chunk_size, device=device).unsqueeze(0)  # [1, T]
            rtc_mask = token_indices < delay.unsqueeze(1)  # [B, T]

            # (c) Per-token time: frozen tokens = 0 (clean, PI05 convention), others = sampled t
            # frozen prefix 被设为 clean action，这样 suffix token 在 attention/mixer 中能看到真实连续的前缀。
            # suffix 仍然使用普通 flow-matching 的 sampled time，保持原始去噪训练目标。
            time_per_token = torch.where(
                rtc_mask, torch.zeros_like(time.unsqueeze(1)), time.unsqueeze(1)
            )  # [B, T]

            # (d) Interpolate with per-token time:
            # At time=0 (frozen): x_t = 0*noise + 1*actions = actions (clean) — automatic
            # At time=t (non-frozen): x_t = t*noise + (1-t)*actions (normal interpolation)
            # 这里不需要手动 copy actions 到 prefix：只要 time_per_token=0，
            # 线性插值公式自然会把 frozen prefix 变成干净动作。
            time_expanded = time_per_token.unsqueeze(-1)  # [B, T, 1]
            x_t = time_expanded * noise + (1 - time_expanded) * actions  # torch.Size([8, 50, 32])
            u_t = noise - actions

            # (e) Effective time for model forward: [B, T] per-token
            # effective_time 传给 suffix embedding / AdaRMS conditioning。
            # 这让模型在同一次 forward 里知道：前缀 token 已经 clean，后缀 token 还处于时间 t。
            effective_time = time_per_token

            # Store loss mask for PI05Policy.forward(); 1=active (loss counted), 0=frozen (masked out)
            # frozen prefix 是条件，不是监督目标；如果也对它算 loss，
            # 模型会被迫在“答案已经给出”的位置继续拟合速度场，反而污染 suffix 学习。
            self._training_rtc_mask = (~rtc_mask).unsqueeze(-1).float()  # [B, T, 1]
        else:
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions     # actions shape torch.Size([2, 50, 32])
            effective_time = time  # [B]
            self._training_rtc_mask = None

        # embed_prefix():
        # - prefix_embs: 图像 + 语言提示的 token embeddings
        # - prefix_pad_masks: 哪些 prefix token 是有效 token
        # - prefix_att_masks: prefix token 的注意力结构标记
        # embed_prefix() returns the observation-side prefix tokens and their masks.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        # embed_suffix():
        # - suffix_embs: 当前 noisy action chunk 对应的 suffix embeddings
        # - suffix_pad_masks / suffix_att_masks: suffix token 的 padding / attention 掩码
        # - adarms_cond: 提供给 action expert 的时间条件向量
        # embed_suffix() returns the action-side suffix tokens and the AdaRMS conditioning.
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, effective_time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 把 prefix 和 suffix 两侧的 mask 拼成整条 token 序列的 mask。
        # Combine prefix and suffix masks into one full-sequence mask description.
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # make_att_2d_masks() 的作用：把 1D padding/attention 标记展开成 token-to-token 的二维可见性矩阵。
        # 为什么需要它：transformer 需要知道“第 i 个 token 能否看见第 j 个 token”。
        # 它怎么做到：对 att_masks 做 cumsum 构造前缀/因果分组，再和 pad_masks 结合去掉 padding。
        # make_att_2d_masks() builds the 2D token visibility matrix used by attention.
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # position_ids 的作用：给整条 prefix+suffix 序列分配位置索引，padding 不计入有效位置。
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # _prepare_attention_masks_4d() 的作用：把二维可见性矩阵变成 HF transformer 能直接消费的 4D mask。
        # 它怎么做到：扩维到 [B,1,Q,K]，并把不可见位置替换成大负值。
        # _prepare_attention_masks_4d() converts the 2D visibility matrix into HF-style 4D attention masks.
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            # self.paligemma_with_expert.forward() 的作用：
            # 真正执行 prefix+suffix 联合 transformer 前向。
            # 为什么这里需要它：前面的 embed_* 只负责把输入变成 token embeddings，
            # 真正的信息融合仍然要靠这一步 transformer 计算。
            # 它怎么做到：把 prefix/suffix embeddings、attention mask、position ids、
            # AdaRMS 条件一起送进双路径模型，返回 suffix 路径隐藏状态。
            # self.paligemma_with_expert.forward() is the actual joint transformer pass.
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        # 这里真正跑一次 joint transformer 前向：
        # prefix 提供观测上下文，suffix 提供动作 token，输出的 suffix_out 是动作路径的隐藏状态。
        # This call runs the joint prefix+suffix transformer and returns the suffix hidden states.
        # _apply_checkpoint() 的作用：在训练阶段按需把这段大前向包进 gradient checkpoint，
        # 用“多算一点换显存”的方式降低内存占用；不开启时就直接正常调用。
        # _apply_checkpoint() optionally wraps the call in gradient checkpointing.
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]  # suffix shape torch.Size([2, 50, 1024])
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            # self.action_out_proj(...) 的作用：
            # 把 expert 隐藏状态投影回动作向量空间。
            # 为什么需要它：transformer 内部宽度是 expert hidden size，不等于真实动作维度。
            # 它怎么做到：通过一个线性层把 hidden_size -> max_action_dim。
            # self.action_out_proj(...) maps hidden states back to action dimensions.
            return self.action_out_proj(suffix_out)

        # 这里得到 v_t，也就是模型预测的 velocity / flow field。
        # 训练目标会让它去逼近 u_t = noise - actions。
        # This projects suffix hidden states into the predicted velocity field v_t.
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        # 这是 chunk 级推理入口：从噪声开始迭代去噪，最终生成整个动作块。
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            # 这里同样从标准高斯噪声开始，只不过是在推理阶段作为去噪初值。
            # At inference, Gaussian noise serves as the initial latent action chunk.
            noise = self.sample_noise(actions_shape, device)

        # 先把 observation 侧输入编码成 prefix token，后续整个去噪过程中都会复用这份前缀上下文。
        # Encode observation inputs once into prefix tokens; the denoising loop reuses this context.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        # 这里只对 prefix 单独建注意力图，因为第一步先缓存 observation 侧上下文。
        # Build a prefix-only attention map because the first pass caches observation context only.
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # 这里先只跑 prefix 路径，并缓存 past_key_values。
        # 后续每一步 denoise_step 都能复用这份 prefix cache，避免重复计算图像/语言前缀。
        # Run the prefix-only pass once and cache past_key_values for all later denoising steps.
        # 这个 self.paligemma_with_expert.forward(...) 调用只计算 prefix，并返回 KV cache。
        # 后面每个 denoise step 都会复用它，因此推理会快很多。
        # This prefix-only call produces the KV cache reused across all denoising steps.
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        # Training-Time RTC inference: get prefix info from kwargs
        # 从上层拿到上一块动作的尾部和延迟长度，用于当前 chunk 的冻结前缀。
        #
        # training_rtc_prev_chunk 的语义：
        # 它不是额外的 guidance target，而是当前 denoising 输入里“已经确定”的 clean prefix。
        # 推理时把上一块 chunk 尚未执行完、需要保持连续的动作放到当前 chunk 开头；
        # 训练时模型已经见过这种“prefix clean + suffix noisy”的分布，所以这里可以直接普通前向，
        # 不必像传统 RTC 那样每步做 VJP/pinv correction。
        training_rtc_prev_chunk = kwargs.get("training_rtc_prev_chunk")
        training_rtc_delay = kwargs.get("training_rtc_delay", 0)

        # Pre-compute per-token mask for RTC (constant across denoising steps)
        # 这一掩码在整个去噪过程中保持不变，表示哪些 token 属于冻结 prefix。
        training_rtc_active = (
            self.config.training_rtc and training_rtc_prev_chunk is not None and training_rtc_delay > 0
        )
        if training_rtc_active:
            rtc_token_mask = (
                torch.arange(self.config.chunk_size, device=device).unsqueeze(0) < training_rtc_delay
            )  # [1, T]

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            if training_rtc_active:
                # Hard-replace prefix x_t with previous chunk's actions (clean)
                # 直接用上一块的干净动作覆盖当前块前缀，这就是 training-time RTC 推理的核心。
                # 这个替换发生在每个 denoise step 的模型输入前：
                # - prefix 始终作为 clean 条件提供给模型；
                # - suffix 保持正常 Euler 去噪状态；
                # - 模型学到的是“看着确定前缀续写后缀”，而不是“整块动作重新发明一遍”。
                x_t = torch.where(rtc_token_mask.unsqueeze(-1), training_rtc_prev_chunk, x_t)
                # Build per-token time: frozen = 0 (clean, PI05 convention), others = current t
                # 冻结 token 的时间设成 0，表示它已经是 clean action。
                # 再强调一次时间约定：PI05 这里 time=0 是 clean，time=1 是 noise；
                # 所以这和官方 kinetix demo 里 frozen time=1 的写法表面不同，但语义一致。
                time_tensor = torch.full(
                    (bsize, self.config.chunk_size), time, dtype=torch.float32, device=device
                )
                time_tensor = torch.where(
                    rtc_token_mask.expand(bsize, -1), torch.zeros_like(time_tensor), time_tensor
                )  # [B, T]
            else:
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                # denoise_step(...) 的作用：
                # 给定当前 x_t 和当前时间，预测一步 velocity v_t。
                # 为什么这里包成 partial call：RTCProcessor 需要一个“原始去噪器”回调，
                # 既可以直接调用，也可以在外面加 guidance。
                # denoise_step(...) predicts one denoising velocity step for the current x_t.
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                # self.rtc_processor.denoise_step(...) 的作用：
                # 在原始 denoise_step 外面再套一层 RTC guidance /约束逻辑。
                # 为什么需要它：传统 RTC 需要根据执行延迟和上一块剩余动作来修正当前去噪结果。
                # 它怎么做到：接收原始 denoise 回调 original_denoise_step_partial，
                # 再结合 RTC 参数输出一个修正后的 v_t。
                # RTCProcessor can wrap the original denoiser with RTC-specific guidance.
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                # 没开 RTC 时，直接走原始 denoise_step。
                v_t = denoise_step_partial_call(x_t)

            # Euler-style 更新：x_{t+dt} = x_t + dt * v_t。
            # 每一步都把当前 latent action chunk 往更“干净”的方向推一点。
            # Euler integration step that updates the latent action chunk using v_t.
            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # 单步去噪：构造 suffix token，再与 prefix cache 一起前向，预测速度场 v_t。
        # embed_suffix() 在这里把“当前时刻的噪声动作 x_t + 时间条件 timestep”
        # 变成动作 expert 可消费的 suffix token。
        # Here embed_suffix() converts the current noisy action and timestep into suffix tokens.
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # 这块 mask 表示“suffix token 能看到哪些 prefix token”。
        # 因为 prefix 是条件上下文，所以这里默认全部开放可见。
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        # 这块 mask 表示 suffix token 之间自己的可见性关系（包含因果结构）。
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        # 把 prefix 可见性和 suffix 内部可见性拼成完整的 suffix 查询图。
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # suffix 的位置编号要接在 prefix 后面，所以先加上 prefix 的有效长度偏移。
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # 深拷贝 past_key_values，避免当前步的 forward 改写缓存，影响下一步复用。
        past_key_values = copy.deepcopy(past_key_values)
        # 这里只跑 suffix 路径，同时借助 prefix 的 KV cache 提供观测上下文。
        # The suffix-only forward uses cached prefix context from past_key_values.
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # outputs_embeds[1] 就是 suffix 路径的隐藏状态；prefix 路径这里没有重新算输出。
        suffix_out = outputs_embeds[1]
        # 只保留动作 chunk 对应的最后这段输出。
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        # 最后再通过 action_out_proj 把 hidden states 投回动作空间，得到 v_t。
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""
    # 这是框架层暴露出去的 policy 包装器：
    # 负责预处理输入、调用底层 PI05Pytorch、管理 action queue 和 RTC 状态。

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        
        中文说明：
            config 是 pi05 的统一配置对象，里面已经包含特征定义、RTC 选项、
            训练相关超参数和模型结构选项。
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        # 先初始化 RTC processor，再构造底层模型，保证模型能拿到 RTC 配置。
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        # 这里重载 from_pretrained，主要是为了兼容 OpenPI 原始权重命名和当前端口实现之间的差异。
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        # 把历史 checkpoint 或 OpenPI 风格命名映射到当前实现参数名。
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        # 清空逐步执行动作的队列，以及 training-time RTC 推理缓存的上一块动作。
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # Training-Time RTC: track previous action chunk for prefix conditioning at inference
        self._prev_action_chunk = None

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        # 传统 RTC guidance 需要 RTCProcessor；training-time RTC 本身并不强依赖它。
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        
        中文说明：
        这里会统一处理图像格式、分辨率和数值范围，产出给 SigLIP/PaliGemma 使用的图像列表和 mask。
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            # img from batch is usually [B, C, H, W], but some pipelines may already provide [B, H, W, C].

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)
                # img: [B, H, W, C]

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
                # After resize/pad, img stays [B, H_resized, W_resized, C]

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                # img back to [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            # mask: [B], one validity flag per camera stream in the current batch
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        # images: list[[B, C, H, W] or [B, H, W, C] after normalization pipeline ends as [B, C, H, W]]
        # img_masks: list[[B]]
        return images, img_masks

    def prepare_action(self, batch):
        """
        Pad action.
        把动作补齐到模型内部固定维度。

        Why:
            PI0.5 internally assumes a fixed action width ``max_action_dim``,
            but a specific robot/dataset may expose a smaller true action dimension.
            PI0.5 内部投影层假设动作维度固定为 ``max_action_dim``，
            但具体机器人或数据集的真实动作维度可能更小。

        How:
            This method calls ``pad_vector(...)`` and appends zeros on the last
            dimension until the action tensor reaches ``max_action_dim``.
            这个函数通过 ``pad_vector(...)`` 在最后一维补 0，
            直到动作张量达到 ``max_action_dim``。
        """
        # pad_vector(...) 的作用：
        # 如果真实动作维度小于 max_action_dim，就在最后一维补 0；
        # 如果已经够大，则直接原样返回。
        # pad_vector(...) either zero-pads to max_action_dim or returns the tensor unchanged.
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        # batch[ACTION]: [B, T, A_real] -> actions: [B, T, A_pad=max_action_dim]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        # 普通同步控制入口：内部仍然按 chunk 预测，再通过队列逐步吐出单步动作。
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            # predict_action_chunk() 先生成整个 horizon；
            # 这里只截前 n_action_steps 个动作放进队列，后续控制循环逐个取出执行。
            # Generate a full chunk first, then queue only the next executable steps.
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # actions after slice: [B, n_action_steps, A_real]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        # popleft() returns one control step: [B, A_real]
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # chunk 推理入口，也是 RTC / training-time RTC 真正能生效的入口。
        self.eval()

        # Prepare inputs
        # _preprocess_images() 会把 batch 里的图像统一成模型期望的格式：
        # 调整分辨率、把数值从 [0,1] 映射到 [-1,1]、并构造每个相机的有效性 mask。
        # _preprocess_images() standardizes images and builds camera-validity masks.
        images, img_masks = self._preprocess_images(batch)
        # images: list[[B, C, H, W]], img_masks: list[[B]]
        # 这里直接取 processor 已经准备好的 language token 和 attention mask。
        # Language tokens and masks are expected to be prepared earlier by the processor pipeline.
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        # tokens: [B, N_text], masks: [B, N_text]

        # Training-Time RTC inference:
        # When training_rtc=True and chunk_size > n_action_steps (temporal overlap), the
        # previous chunk's unexecuted tail is used as a frozen prefix for the new chunk.
        # This is the core benefit of Training-Time RTC: prefix hard-replacement + per-token
        # time, NO expensive pinv/VJP guidance — just a regular forward pass.
        #
        # The prefix length is capped at simulated_delay - 1 to stay within the training
        # distribution (training only sees delays in {0, ..., simulated_delay - 1}).
        # When chunk_size == n_action_steps (no overlap), this block is skipped.
        #
        # 为什么用上一块 chunk 的尾部：
        # action chunk 通常一次预测 T 步，但控制循环可能只执行前 n_action_steps 步就重新规划。
        # 如果 T > n_action_steps，那么上一块的 [n_action_steps : T) 仍然描述了未来一段动作。
        # 把这段尾部移到下一块开头，相当于告诉模型：
        # “这些动作已经由上一次计划确定了，请从这里自然续写。”
        #
        # 为什么 cap 到 simulated_delay - 1：
        # 训练只见过 0 到 simulated_delay-1 个 frozen token。
        # 推理时如果塞入更长 prefix，会把模型推到没训练过的条件分布外。
        if self.config.training_rtc and self._prev_action_chunk is not None:
            overlap = self.config.chunk_size - self.config.n_action_steps
            inference_delay = min(overlap, self.config.simulated_delay - 1)
            if inference_delay > 0:
                shift = self.config.n_action_steps
                shifted_prev = torch.zeros_like(self._prev_action_chunk)
                # shifted_prev 的前 inference_delay 个 token 对齐到当前 chunk 开头；
                # 其它 token 保持 0 只是占位，因为 sample_actions 只会根据 training_rtc_delay
                # 对前缀位置做 hard replacement。
                shifted_prev[:, :inference_delay] = self._prev_action_chunk[:, shift : shift + inference_delay]
                kwargs["training_rtc_prev_chunk"] = shifted_prev
                kwargs["training_rtc_delay"] = inference_delay

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        # PI0.5 不需要像某些策略那样单独维护额外 state 输入，直接走 sample_actions 即可。
        # sample_actions() 的做法是：先编码 prefix，再从噪声出发做多步去噪，
        # 最终输出整个动作 chunk。
        # sample_actions() encodes the prefix once, then iteratively denoises a whole action chunk.
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
        # actions from the core model: [B, T=chunk_size, A_pad]

        # Store full horizon chunk for next call's prefix (only if user opts into inference RTC)
        # 这里缓存的是 padded action chunk，和模型内部 action dim 对齐。
        # 下一次构造 prefix 时也发生在模型内部 padded 空间里，因此先缓存再 unpad。
        if self.config.training_rtc:
            self._prev_action_chunk = actions.detach().clone()

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        # returned actions: [B, T=chunk_size, A_real]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        
        中文说明：
        这是训练入口，不是推理入口。它会先做 batch 整理，再调用底层网络返回逐元素损失，
        最后在这里完成动作维度截断、RTC mask 处理和 loss 归约。
        """
        # Prepare inputs
        # _preprocess_images() 负责把原始图像 batch 变成 PI0.5 底层网络可直接消费的图像列表和 mask。
        # It converts raw image tensors into model-ready image inputs plus camera masks.
        images, img_masks = self._preprocess_images(batch) # images: list[[B, C, H, W]] len为3,表示三个相机, img_masks: list[[B]] 
        # images: list[[B, C, H, W]], img_masks: list[[B]]
        # token/mask 在这里直接来自 processor_pi05 的 tokenizer 输出。
        # These text tokens/masks come directly from the pi05 processor tokenizer output.
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        # tokens: [B, N_text]torch.Size([2, 200]), masks: [B, N_text]  

        # prepare_action() 的作用是把真实动作维度补齐到模型内部固定维度。
        # 做法很简单：对最后一维补 0，直到达到 max_action_dim。
        # prepare_action() pads the robot-specific action vector with zeros to max_action_dim.
        actions = self.prepare_action(batch)   # batch[ACTION]: [B, T, A_real] -> actions: [B, T, A_pad], shape:[2,50,12] ->torch.Size([2, 50, 32])
        # actions: [B, T=chunk_size, A_pad]

        # Compute loss (no separate state needed for PI05)
        # 底层 model.forward() 返回的是逐 token、逐动作维度的未归约损失张量，
        # 这样上层才能继续做 RTC mask 和不同 reduction 策略。
        # The core model returns unreduced per-token/per-dimension losses.
        losses = self.model.forward(images, img_masks, tokens, masks, actions)
        # losses before truncation: [B, T=chunk_size, A_pad]

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        # losses after truncation: [B, T=chunk_size, A_real]

        # Training-Time RTC: apply masked loss (only on non-frozen tokens)
        rtc_mask = self.model._training_rtc_mask if self.config.training_rtc else None

        if rtc_mask is not None:
            # rtc_mask: [B, T, 1], 1=active, 0=frozen; truncate action dim to match
            # 只有非冻结 token 会参与损失，这正是 training-time RTC 的监督方式。
            #
            # 原理：
            # frozen prefix 在输入 x_t 里已经被设置成 clean action，它的角色是“条件”。
            # 如果继续在这些位置监督 velocity，相当于要求模型对已知答案再做预测，
            # 这会把训练信号浪费在 prefix 上，并可能干扰模型学习 suffix 如何接上 prefix。
            # 所以这里所有统计和最终 loss 都只看 rtc_mask=1 的 active suffix token。
            loss_dict = {
                "loss_per_dim": (losses * rtc_mask).sum(dim=[0, 1]).detach().cpu()
                / (rtc_mask.sum(dim=[0, 1]).clamp(min=1e-8)).detach().cpu(),
            }
            loss_dict["loss_per_dim"] = loss_dict["loss_per_dim"].numpy().tolist()

            if reduction == "none":
                # Per-sample masked mean over time and action dims
                # reduction="none" 用于上层按样本重新加权，例如 RA-BC/DSRL 类训练。
                # 因此这里保留每个样本一个 loss，但仍然只平均 active suffix token。
                per_sample_loss = (losses * rtc_mask).sum(dim=(1, 2)) / (
                    rtc_mask.sum(dim=(1, 2)) * original_action_dim + 1e-8
                )
                # per_sample_loss: [B]
                loss_dict["loss"] = per_sample_loss.mean().item()
                return per_sample_loss, loss_dict
            else:
                loss = (losses * rtc_mask).sum() / (rtc_mask.sum() * original_action_dim + 1e-8)
                loss_dict["loss"] = loss.item()
                return loss, loss_dict
        else:
            # ---- Original path (unchanged) ----
            # 没启用 training-time RTC 时，退回标准平均 MSE 路径。
            loss_dict = {
                "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            }

            if reduction == "none":
                # Return per-sample losses (B,) by averaging over time and action dims
                per_sample_loss = losses.mean(dim=(1, 2))
                # per_sample_loss: [B]
                loss_dict["loss"] = per_sample_loss.mean().item()
                return per_sample_loss, loss_dict
            else:
                # Default: return scalar mean loss
                loss = losses.mean()
                loss_dict["loss"] = loss.item()
                return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        # 默认 PEFT 目标集中在 action expert 的注意力投影和动作相关投影层。
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
