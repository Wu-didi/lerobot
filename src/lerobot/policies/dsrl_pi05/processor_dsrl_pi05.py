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

from typing import Any

import torch

from lerobot.policies.dsrl_pi05.configuration_dsrl_pi05 import DSRLPi05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_dsrl_pi05_pre_post_processors(
    config: DSRLPi05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """DSRL pi0.5 复用 pi0.5 的观测和动作 processor，确保输入输出归一化完全一致。

    DSRL 只改变 policy 内部的 latent-noise 选择方式，不改变数据格式；
    因此 processor 不能另起一套，否则 offline cache 和 base pi0.5 解码器会对不齐。
    """

    return make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
