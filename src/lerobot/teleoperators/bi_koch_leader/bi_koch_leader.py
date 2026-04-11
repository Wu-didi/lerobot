#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from functools import cached_property

from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
from lerobot.teleoperators.koch_leader.koch_leader import KochLeader

from ..teleoperator import Teleoperator
from .config_bi_koch_leader import BiKochLeaderConfig

logger = logging.getLogger(__name__)


class BiKochLeader(Teleoperator):
    config_class = BiKochLeaderConfig
    name = "bi_koch_leader"

    def __init__(self, config: BiKochLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = KochLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            gripper_open_pos=config.left_arm_gripper_open_pos,
        )

        right_arm_config = KochLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            gripper_open_pos=config.right_arm_gripper_open_pos,
        )

        self.left_arm = KochLeader(left_arm_config)
        self.right_arm = KochLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Remove "left_" prefix
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        self._disconnect_arm("left", self.left_arm)
        self._disconnect_arm("right", self.right_arm)

    def _split_prefixed_positions(
        self, positions: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        left_positions = {
            key.removeprefix("left_").removesuffix(".pos"): value
            for key, value in positions.items()
            if key.startswith("left_") and key.endswith(".pos")
            and key.removeprefix("left_").removesuffix(".pos") != "gripper"
        }
        right_positions = {
            key.removeprefix("right_").removesuffix(".pos"): value
            for key, value in positions.items()
            if key.startswith("right_") and key.endswith(".pos")
            and key.removeprefix("right_").removesuffix(".pos") != "gripper"
        }
        return left_positions, right_positions

    def write_goal_positions(self, positions: dict[str, float]) -> None:
        left_positions, right_positions = self._split_prefixed_positions(positions)

        if left_positions:
            self.left_arm.bus.sync_write("Goal_Position", left_positions)
        if right_positions:
            self.right_arm.bus.sync_write("Goal_Position", right_positions)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.write_goal_positions(action)
        return action

    def enable_torque(self):
        self.left_arm.bus.enable_torque(self._body_motors(self.left_arm))
        self.right_arm.bus.enable_torque(self._body_motors(self.right_arm))

    def disable_torque(self):
        self._release_for_manual_control(self.left_arm, self.config.left_arm_gripper_open_pos)
        self._release_for_manual_control(self.right_arm, self.config.right_arm_gripper_open_pos)

    def _body_motors(self, arm: KochLeader) -> list[str]:
        return [motor for motor in arm.bus.motors if motor != "gripper"]

    def _hold_gripper_open(self, arm: KochLeader, open_pos: float) -> None:
        arm.bus.enable_torque("gripper", num_retry=1)
        if arm.is_calibrated:
            arm.bus.write("Goal_Position", "gripper", open_pos, num_retry=1)

    def _release_for_manual_control(self, arm: KochLeader, gripper_open_pos: float) -> None:
        self._hold_gripper_open(arm, gripper_open_pos)
        arm.bus.disable_torque(self._body_motors(arm), num_retry=1)

    def _disconnect_arm(self, side: str, arm: KochLeader) -> None:
        if not arm.is_connected:
            return

        if self.config.disable_torque_on_disconnect:
            try:
                arm.bus.disable_torque("gripper", num_retry=5)
            except Exception as exc:
                logger.warning(
                    "Failed to disable %s leader gripper while disconnecting. Error: %s",
                    side,
                    exc,
                )
            try:
                arm.bus.disable_torque(self._body_motors(arm), num_retry=5)
            except Exception as exc:
                logger.warning(
                    "Failed to disable %s leader body motors while disconnecting. Error: %s",
                    side,
                    exc,
                )

        try:
            arm.bus.disconnect(disable_torque=False)
            logger.info(f"{arm} disconnected.")
        except Exception as exc:
            logger.warning("Failed to disconnect %s leader cleanly. Error: %s", side, exc)
