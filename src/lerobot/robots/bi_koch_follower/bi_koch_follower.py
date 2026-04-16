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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.koch_follower import KochFollower
from lerobot.robots.koch_follower.config_koch_follower import KochFollowerConfig

from ..robot import Robot
from .config_bi_koch_follower import BiKochFollowerConfig

logger = logging.getLogger(__name__)


class BiKochFollower(Robot):
    """双臂 Koch follower 机器人实现。

    这个类本质上不是“重新实现一套全新的双臂控制逻辑”，而是做了一层组合封装：
    - 左臂：一个 `KochFollower`
    - 右臂：一个 `KochFollower`
    - 相机：一组独立 camera

    它最核心的设计点是“名字空间前缀”：
    - 左臂所有 joint / action / observation 键名前面都加 `left_`
    - 右臂所有键名前面都加 `right_`

    这样上层看到的就是一个统一的双臂 robot：
    - observation 是一份大字典
    - action 也是一份大字典
    - 但内部仍然可以很清楚地拆回左右两套单臂对象
    """
    config_class = BiKochFollowerConfig
    name = "bi_koch_follower"

    def __init__(self, config: BiKochFollowerConfig):
        """根据双臂配置构造左右两只单臂 follower，并初始化共享相机。

        作用：
        1. 保存原始双臂配置
        2. 把双臂配置拆成左右两份 `KochFollowerConfig`
        3. 分别构造 `self.left_arm` 与 `self.right_arm`
        4. 构造挂在整机上的相机集合

        这里的关键点是：双臂类本身并不直接管理电机细节，而是把单臂类复用起来。
        所以双臂逻辑更多是在做“配置拆分”和“键名拼接/拆解”。
        """
        super().__init__(config)
        self.config = config

        # 先把双臂配置拆成左臂的单臂配置。
        # 注意这里把 `config.id` 自动扩展成 `<id>_left`，这样左右臂的标定文件、
        # 标识符和日志上下文都不会混淆。
        left_arm_config = KochFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )

        # 同理，构造右臂的单臂配置。
        right_arm_config = KochFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        # 双臂机器人内部实际是“两个单臂对象 + 一组共享相机”。
        self.left_arm = KochFollower(left_arm_config)
        self.right_arm = KochFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """返回双臂电机关节的 feature 定义。

        作用：
        - 生成 action / observation 里电机相关字段的 schema
        - 给每个键统一加上 `left_` / `right_` 前缀

        例如如果单臂有 `shoulder.pos`，这里会变成：
        - `left_shoulder.pos`
        - `right_shoulder.pos`
        """
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """返回相机相关的 feature 定义。

        每个相机键对应一个 `(H, W, 3)` 的图像 shape，用于描述 observation
        schema。这里直接从配置里读取相机分辨率，而不是在运行时从图像帧推断。
        """
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """双臂机器人的 observation schema。

        组成：
        - 左右电机状态字段
        - 所有相机图像字段

        上层 recorder / dataset / policy 会依赖这份定义来推导 observation 的
        键名和类型。
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """双臂机器人的 action schema。

        对 follower 来说，action 只包含关节目标，不包含相机，所以这里直接返回
        电机字段定义。
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """判断整台双臂机器人是否处于“完全连接”状态。

        判定条件：
        - 左臂总线已连接
        - 右臂总线已连接
        - 所有相机都已连接

        只有三者都满足，才认为整个双臂对象已经 ready。
        """
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """连接左右臂和所有相机。

        作用：
        - 先连接左臂
        - 再连接右臂
        - 最后连接所有 camera

        参数 `calibrate` 会透传给单臂 `connect()`，决定连接时是否顺带检查/执行标定。
        """
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        """判断左右两只手臂是否都完成标定。"""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """依次触发左右臂的标定流程。"""
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        """依次执行左右臂的运行前配置。

        这里复用单臂的 `configure()`，双臂类本身不额外引入新的配置逻辑。
        """
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        """依次执行左右臂的电机初始化。"""
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        """读取双臂 + 相机的完整 observation。

        返回值是一份统一字典，内部包含三部分：
        1. 左臂 observation，所有键加 `left_` 前缀
        2. 右臂 observation，所有键加 `right_` 前缀
        3. 相机图像，保持原相机键名

        这样上层调用者不需要分别管理左右两只手臂，只需要面对一个整机 observation。
        """
        obs_dict = {}

        # 先读取左臂 observation，并统一加上 `left_` 前缀，避免和右臂键名冲突。
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # 再读取右臂 observation，并加上 `right_` 前缀。
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # 最后读取所有相机。
        # `max_age_ms=1000` 的意思是：允许读取最多 1 秒内的最新帧，
        # 避免相机短暂掉帧时直接拿不到图像。
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest(max_age_ms=1000)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """向左右两臂分别下发动作，并返回实际发送出去的动作。

        输入：
        - 一份双臂 action 字典，键名使用 `left_` / `right_` 前缀

        内部流程：
        1. 把大字典按前缀拆成 left_action / right_action
        2. 分别调用左右单臂的 `send_action`
        3. 把单臂返回结果重新加前缀，拼成一份双臂字典返回

        返回值通常用于上层记录“真正发给机器人的动作”，而不只是用户请求的动作。
        """
        # 从总 action 里筛出左臂动作，并把 `left_` 前缀去掉，恢复成单臂接口需要的键名。
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }

        # 同理，拆出右臂动作。
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        # 分别把动作下发给左右单臂。
        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # 单臂返回的动作键名不带前缀，这里重新补回前缀，恢复成双臂统一视图。
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        """断开左右臂与所有相机的连接。

        顺序上先断手臂，再断相机。这里同样只是做组合转发，不额外加入新的资源管理逻辑。
        """
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
