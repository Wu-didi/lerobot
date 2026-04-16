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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_koch_follower")
@dataclass
class BiKochFollowerConfig(RobotConfig):
    """双臂 Koch follower 机器人的配置类。

    这个配置类本身并不直接驱动硬件，它的作用是把“构造双臂机器人所需的参数”
    组织成一份结构化配置，供 `BiKochFollower` 在初始化时消费。

    可以把它理解成：
    - 左臂一份单臂 KochFollower 的必要参数
    - 右臂一份单臂 KochFollower 的必要参数
    - 再额外挂一组共享相机配置

    后续 `BiKochFollower.__init__()` 会把这里的字段拆成左右两套
    `KochFollowerConfig`，然后分别构造 `self.left_arm` 和 `self.right_arm`。
    """
    # 左右两只 follower 机械臂的串口。
    # 这是创建底层电机总线时最核心的硬件定位信息。
    left_arm_port: str
    right_arm_port: str

    # 左臂可选配置：
    # - disconnect 时是否自动关力矩
    # - 是否允许相对动作裁剪
    # - 动作单位是否使用角度制
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: int | None = None
    left_arm_use_degrees: bool = False

    # 右臂可选配置，含义与左臂完全对称。
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: int | None = None
    right_arm_use_degrees: bool = False

    # 共享相机配置。
    # 这里的 camera 不属于某一只手臂内部，而是挂在整个双臂机器人对象上。
    # 例如俯视相机、左右腕部相机，都通过这个字典统一传入。
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
