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

from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.dsrl_pi05.configuration_dsrl_pi05 import DSRLPi05Config
from lerobot.policies.dsrl_pi05.modeling_dsrl_pi05 import DSRLPi05Policy
from lerobot.utils.constants import ACTION, OBS_STATE


class DummyBasePI05Policy(nn.Module):
    def __init__(self, chunk_size: int, max_action_dim: int, action_dim: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            chunk_size=chunk_size,
            max_action_dim=max_action_dim,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            },
        )


def make_test_config() -> DSRLPi05Config:
    config = DSRLPi05Config(
        base_policy_path=None,
        device="cpu",
        chunk_size=3,
        n_action_steps=2,
        max_action_dim=4,
        observation_feature_dim=6,
        policy_hidden_dims=[8],
        latent_inversion_steps=120,
        latent_restarts=1,
        latent_inversion_lr=0.2,
        latent_reg_weight=0.0,
        noise_action_magnitude=2.0,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    return config


def test_dsrl_pi05_forward_select_and_invert():
    config = make_test_config()
    policy = DSRLPi05Policy(
        config=config,
        base_policy=DummyBasePI05Policy(chunk_size=3, max_action_dim=4, action_dim=2),
    )
    assert not next(policy.base_policy.parameters()).requires_grad

    policy._encode_observation_context = lambda batch: batch["features"]  # type: ignore[method-assign]
    policy.decode_noise_to_actions = (  # type: ignore[method-assign]
        lambda batch, noise, num_steps=None: policy._reshape_noise(noise)[:, :, :2]
    )
    policy._build_prefix_cache = (  # type: ignore[method-assign]
        lambda batch: (torch.ones(batch["features"].shape[0], 1, dtype=torch.bool), None)
    )
    policy._decode_noise_from_prefix_cache = (  # type: ignore[method-assign]
        lambda prefix_pad_masks, past_key_values, noise, num_steps=None: policy._reshape_noise(noise)[:, :, :2]
    )
    policy.set_return_to_go_stats(0.0, 1.0)

    batch = {
        "features": torch.randn(2, 6),
        "noise_label": torch.randn(2, config.latent_action_dim),
        "return_to_go": torch.tensor([-1.0, 0.5]),
    }
    loss, loss_dict = policy.forward(batch)
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert loss_dict["mean_weight"] >= 1.0

    action_chunk = policy.predict_action_chunk({"features": torch.randn(2, 6)})
    assert action_chunk.shape == (2, config.chunk_size, 2)

    first_action = policy.select_action({"features": torch.randn(1, 6)})
    second_action = policy.select_action({"features": torch.randn(1, 6)})
    assert first_action.shape == (1, 2)
    assert second_action.shape == (1, 2)

    target_actions = torch.randn(2, config.chunk_size, 2) * 0.1
    noise_label, recon_error = policy.invert_actions_to_noise({"features": torch.randn(2, 6)}, target_actions)
    assert noise_label.shape == (2, config.chunk_size, config.max_action_dim)
    assert recon_error.shape == (2,)
    assert torch.max(recon_error).item() < 1e-4


def test_dsrl_pi05_iql_losses():
    config = make_test_config()
    config.offline_algorithm = "iql"
    policy = DSRLPi05Policy(
        config=config,
        base_policy=DummyBasePI05Policy(chunk_size=3, max_action_dim=4, action_dim=2),
    )

    obs_features = torch.randn(4, 6)
    next_obs_features = torch.randn(4, 6)
    behavior_noise = torch.randn(4, config.chunk_size, config.max_action_dim) * 0.1
    reward = torch.tensor([-1.0, -1.0, 0.0, -1.0])
    done = torch.tensor([0.0, 0.0, 1.0, 1.0])

    critic_loss, critic_dict = policy.compute_critic_loss(
        obs_features=obs_features,
        behavior_noise=behavior_noise,
        reward=reward,
        next_obs_features=next_obs_features,
        done=done,
    )
    value_loss, value_dict = policy.compute_value_loss(
        obs_features=obs_features,
        behavior_noise=behavior_noise,
    )
    actor_loss, actor_dict = policy.compute_iql_actor_loss(
        obs_features=obs_features,
        behavior_noise=behavior_noise,
    )

    assert critic_loss.ndim == 0
    assert value_loss.ndim == 0
    assert actor_loss.ndim == 0
    assert critic_dict["critic_loss"] >= 0
    assert value_dict["value_loss"] >= 0
    assert actor_dict["mean_weight"] > 0


def test_dsrl_pi05_save_pretrained_skips_base_policy_weights(tmp_path):
    config = make_test_config()
    policy = DSRLPi05Policy(
        config=config,
        base_policy=DummyBasePI05Policy(chunk_size=3, max_action_dim=4, action_dim=2),
    )
    policy.save_pretrained(tmp_path)
    state_dict = load_file(tmp_path / "model.safetensors")

    assert len(state_dict) > 0
    assert all(not key.startswith("base_policy.") for key in state_dict)
