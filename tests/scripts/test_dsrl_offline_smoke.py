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

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.dsrl_pi05.configuration_dsrl_pi05 import DSRLPi05Config
from lerobot.policies.dsrl_pi05.modeling_dsrl_pi05 import DSRLPi05Policy
from lerobot.scripts import (
    lerobot_precompute_dsrl_offline_cache as cache_script,
    lerobot_train_offline_dsrl as train_script,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.fixtures.constants import DUMMY_REPO_ID, DEFAULT_FPS


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


def _make_policy_config() -> DSRLPi05Config:
    return DSRLPi05Config(
        base_policy_path=None,
        device="cpu",
        chunk_size=2,
        n_action_steps=1,
        max_action_dim=2,
        observation_feature_dim=8,
        policy_hidden_dims=[16],
        critic_hidden_dims=[16],
        value_hidden_dims=[16],
        latent_inversion_steps=60,
        latent_restarts=1,
        latent_inversion_lr=0.2,
        latent_reg_weight=0.0,
        latent_recon_threshold=1e-3,
        query_stride=1,
    )


def _make_fake_make_policy():
    def _fake_make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):  # noqa: ARG001
        features = dataset_to_policy_features(ds_meta.features)
        cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
        policy = DSRLPi05Policy(
            cfg,
            base_policy=DummyBasePI05Policy(
                chunk_size=cfg.chunk_size,
                max_action_dim=cfg.max_action_dim,
                action_dim=cfg.output_features[ACTION].shape[0],
            ),
        )

        def _encode_observation_context(batch):
            state = batch[OBS_STATE].to(dtype=torch.float32)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            pad = cfg.observation_feature_dim - state.shape[-1]
            return torch.nn.functional.pad(state, (0, pad))

        policy._encode_observation_context = _encode_observation_context  # type: ignore[method-assign]
        policy._build_prefix_cache = (  # type: ignore[method-assign]
            lambda batch: (
                torch.ones(batch[OBS_STATE].shape[0], 1, dtype=torch.bool, device=batch[OBS_STATE].device),
                None,
            )
        )
        policy._decode_noise_from_prefix_cache = (  # type: ignore[method-assign]
            lambda prefix_pad_masks, past_key_values, noise, num_steps=None: policy._reshape_noise(noise)[
                :, :, : cfg.output_features[ACTION].shape[0]
            ]
        )
        return policy

    return _fake_make_policy


def _make_identity_preprocessor(device: str):
    def _preprocessor(batch):
        processed = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.to(device)
            else:
                processed[key] = value
        return processed

    return _preprocessor


def test_dsrl_offline_scripts_smoke(tmp_path, monkeypatch):
    features = {
        ACTION: {"dtype": "float32", "shape": (2,), "names": None},
        OBS_STATE: {"dtype": "float32", "shape": (4,), "names": None},
    }
    dataset_root = tmp_path / "dataset"
    dataset = cache_script.LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=features,
        root=dataset_root,
        use_videos=False,
    )
    for episode_idx in range(2):
        for step in range(5):
            dataset.add_frame(
                {
                    ACTION: torch.tensor(
                        [episode_idx * 0.1 + step * 0.01, episode_idx * 0.2 + step * 0.02], dtype=torch.float32
                    ),
                    OBS_STATE: torch.tensor(
                        [step, step + 1, episode_idx, 1.0], dtype=torch.float32
                    ),
                    "task": "fold clothes",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()

    fake_make_policy = _make_fake_make_policy()
    identity_preprocessor = _make_identity_preprocessor("cpu")
    monkeypatch.setattr(cache_script, "make_policy", fake_make_policy)
    monkeypatch.setattr(train_script, "make_policy", fake_make_policy)
    monkeypatch.setattr(cache_script, "make_pre_post_processors", lambda policy, dataset_stats=None: (identity_preprocessor, None))
    monkeypatch.setattr(train_script, "make_pre_post_processors", lambda policy, dataset_stats=None: (identity_preprocessor, None))

    policy_cfg = _make_policy_config()
    cache_cfg = cache_script.DSRLOfflineCacheConfig(
        policy=policy_cfg,
        dataset=cache_script.DSRLOfflineDatasetConfig(
            repo_id=DUMMY_REPO_ID,
            root=dataset_root,
            batch_size=2,
            num_workers=0,
        ),
        output_dir=tmp_path / "cache",
        seed=0,
    )
    cache_script.main.__wrapped__(cache_cfg)

    assert (cache_cfg.output_dir / "dsrl_offline_cache.safetensors").exists()
    assert (cache_cfg.output_dir / "dsrl_offline_cache.json").exists()

    train_cfg = train_script.DSRLOfflineTrainConfig(
        policy=policy_cfg,
        cache_dir=cache_cfg.output_dir,
        output_dir=tmp_path / "train",
        batch_size=2,
        num_workers=0,
        steps=3,
        log_freq=1,
        save_freq=2,
        seed=0,
    )
    train_script.main.__wrapped__(train_cfg)

    assert (train_cfg.output_dir / "final" / "model.safetensors").exists()
    assert (train_cfg.output_dir / "final" / "config.json").exists()
    assert (train_cfg.output_dir / "final" / "training_summary.json").exists()
