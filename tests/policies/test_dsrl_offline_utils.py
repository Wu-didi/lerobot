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

import pandas as pd
import torch

from lerobot.policies.dsrl_pi05.offline_utils import (
    build_awr_weights,
    build_episode_candidate_indices,
    build_sparse_reward_table,
    resolve_episode_success_map,
)


class HFDatasetStub:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


class ReaderStub:
    def __init__(self, rows):
        self.hf_dataset = HFDatasetStub(rows)

    def load_and_activate(self):
        return None


def test_sparse_reward_table_and_candidate_selection():
    rows = [
        {"index": torch.tensor(0), "success": torch.tensor(True)},
        {"index": torch.tensor(1), "success": torch.tensor(True)},
        {"index": torch.tensor(2), "success": torch.tensor(True)},
        {"index": torch.tensor(3), "success": torch.tensor(True)},
        {"index": torch.tensor(4), "success": torch.tensor(True)},
        {"index": torch.tensor(5), "success": torch.tensor(False)},
        {"index": torch.tensor(6), "success": torch.tensor(False)},
        {"index": torch.tensor(7), "success": torch.tensor(False)},
        {"index": torch.tensor(8), "success": torch.tensor(False)},
    ]
    episodes = pd.DataFrame(
        [
            {"dataset_from_index": 0, "dataset_to_index": 5},
            {"dataset_from_index": 5, "dataset_to_index": 9},
        ]
    )
    dataset = SimpleNamespace(
        reader=ReaderStub(rows),
        meta=SimpleNamespace(episodes=episodes, total_episodes=2, features={"success": {}}),
        episodes=None,
    )

    candidates = build_episode_candidate_indices(dataset, chunk_size=3, query_stride=2)
    assert candidates == {0: [0, 2], 1: [5]}

    success_map = resolve_episode_success_map(dataset, candidates, success_feature="success")
    assert success_map == {0: True, 1: False}

    reward_table = build_sparse_reward_table(candidates, success_map, discount=0.99)
    assert reward_table["dataset_index"].tolist() == [0, 2, 5]
    assert reward_table["next_dataset_index"].tolist() == [2, -1, -1]
    assert reward_table["reward"].tolist() == [-1.0, 0.0, -1.0]
    assert reward_table["done"].tolist() == [False, True, True]
    assert reward_table["return_to_go"].tolist() == [-1.0, 0.0, -1.0]


def test_awr_weights_are_monotonic():
    return_to_go = torch.tensor([-2.0, 0.0, 1.0], dtype=torch.float32)
    weights = build_awr_weights(return_to_go, beta=1.0, clip=100.0, mean=0.0, std=1.0)
    assert weights[0] <= weights[1] < weights[2]
    assert torch.all(weights >= 1.0)
