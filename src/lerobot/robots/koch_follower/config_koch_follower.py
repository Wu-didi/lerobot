# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


@RobotConfig.register_subclass("koch_follower")
@dataclass
class KochFollowerConfig(RobotConfig):
    """单臂 Koch follower 机器人的配置类。

    这份配置描述的是“如何构造一只 follower 机械臂”：
    - 串口在哪
    - 断连时是否关力矩
    - 是否启用相对动作限幅
    - 是否挂相机
    - 电机归一化时是否使用角度单位

    `KochFollower` 会读取这里的字段来初始化 Dynamixel 电机总线、
    标定信息和相机对象。
    """
    # 连接 follower 机械臂控制板的串口。
    port: str

    # disconnect 时是否自动关闭电机力矩。
    # 对真实机器人来说，这决定了退出程序后机械臂是“保持位置”还是“失去刚度”。
    disable_torque_on_disconnect: bool = True

    # 相对动作安全限幅。
    # 作用：
    # - 防止单步目标位置跳得太远
    # - 降低策略输出异常时对真实硬件造成的风险
    #
    # 支持两种写法：
    # - 一个标量：所有电机共用同样的上限
    # - 一个 dict：给不同电机单独设上限
    max_relative_target: float | dict[str, float] | None = None

    # 挂在这只机械臂上的相机配置。
    # 上层 observation 会把这些图像和电机状态一起返回。
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 是否把关节值按“角度”而不是默认归一化范围输出。
    # 这个主要是为了兼容旧数据集 / 旧策略的表示方式。
    use_degrees: bool = False
