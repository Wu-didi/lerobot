#!/usr/bin/env python

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

import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_koch_follower import KochFollowerConfig

logger = logging.getLogger(__name__)


class KochFollower(Robot):
    """单臂 Koch follower 机器人实现。

    支持的硬件来源：
    - Koch v1.0
    - Koch v1.1

    这个类的职责可以概括为四件事：
    1. 用 `DynamixelMotorsBus` 管理整只机械臂的电机总线
    2. 在 connect / calibrate / configure 阶段完成硬件准备
    3. 在 `get_observation()` 中读取关节状态和相机图像
    4. 在 `send_action()` 中把目标关节位置写入电机

    它是单臂版本；双臂版本 `BiKochFollower` 只是把两个这里的对象组合起来。
    """

    config_class = KochFollowerConfig
    name = "koch_follower"

    def __init__(self, config: KochFollowerConfig):
        """构造单臂 follower 的电机总线和相机对象。

        作用：
        - 保存配置
        - 根据 `use_degrees` 决定电机关节的归一化模式
        - 创建 Dynamixel 总线，并绑定每个关节的电机 id / 型号
        - 创建相机集合

        这里最关键的是 `self.bus`：
        后续几乎所有和机械臂本体相关的操作，最终都会落到这个总线上。
        """
        super().__init__(config)
        self.config = config

        # 身体关节可以选择用角度制或默认归一化范围表示；
        # gripper 则始终单独使用 [0, 100] 风格的归一化。
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
                "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
                "elbow_flex": Motor(3, "xl330-m288", norm_mode_body),
                "wrist_flex": Motor(4, "xl330-m288", norm_mode_body),
                "wrist_roll": Motor(5, "xl330-m288", norm_mode_body),
                "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """返回电机关节的 feature 定义。

        例如：
        - `shoulder_pan.pos`
        - `gripper.pos`

        这些键会被上层 dataset / policy / processor 用来推导 schema。
        """
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """返回相机图像的 feature 定义。

        每个相机键对应 `(H, W, 3)`，表示 RGB 图像的 shape。
        """
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """单臂机器人的 observation schema。

        包含两类字段：
        - 电机状态
        - 相机图像
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """单臂机器人的 action schema。

        对 follower 来说，action 只包含关节目标位置，所以这里只有电机字段。
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """判断机械臂和相机是否都已经连接。"""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """连接机械臂总线和所有相机，并在必要时执行标定。

        工作流程：
        1. 连接 Dynamixel 总线
        2. 如果当前未标定且允许自动标定，则进入 `calibrate()`
        3. 连接所有相机
        4. 执行 `configure()` 设置运行模式

        这里默认假设：连接时机械臂处在一个安全静止姿态，允许短暂关闭力矩做标定。
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """判断电机总线是否已经有有效标定。"""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """执行 follower 机械臂的标定流程。

        作用：
        - 计算每个电机的 homing offset
        - 估计各关节的运动范围
        - 生成 `MotorCalibration`
        - 写回电机并保存到本地校准文件

        大致流程：
        1. 先关力矩，保证手动摆动机械臂是安全的
        2. 如果已有 calibration 文件，允许用户直接复用
        3. 否则进入手动交互式标定
        4. 保存得到的 calibration
        """
        self.bus.disable_torque()
        if self.calibration:
            # 已有 calibration 文件时，允许用户选择：
            # - 直接把旧标定重新写入电机
            # - 或者重新跑一遍标定
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")
        for motor in self.bus.motors:
            # 标定阶段先把所有非夹爪关节都切到 extended position 模式，
            # 避免普通 joint mode 的 0~4095 限制影响多圈关节的中位对齐。
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # shoulder_pan / wrist_roll 是全圈关节，范围直接视为完整一圈；
        # 其他关节需要通过人工摆动来记录最小/最大值。
        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their entire "
            "ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """配置机械臂运行模式和关键控制参数。

        作用：
        - 调用总线级别的默认配置
        - 给大多数关节设置 extended position mode
        - 给 gripper 设置 current-based position mode
        - 微调某些关节的 PID 参数

        这一步是在“设备已经连接、标定也准备好了”之后做的运行时配置。
        """
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            # 除 gripper 外，其余关节都切到 extended position mode。
            # 原因是普通 joint mode 只能覆盖 0~4095，一旦装配时零位不理想，
            # 某些关节可能会在关键姿态附近卡到边界。
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # gripper 使用电流受限的位置控制：
            # 这样即使目标是完全闭合，抓到物体时也不至于过度用力。
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

            # 针对 elbow_flex 调一组更激进的 PID，减小“命令动作”和“实际达到位置”
            # 之间的滞后。
            self.bus.write("Position_P_Gain", "elbow_flex", 1500)
            self.bus.write("Position_I_Gain", "elbow_flex", 0)
            self.bus.write("Position_D_Gain", "elbow_flex", 600)

    def setup_motors(self) -> None:
        """交互式逐个初始化电机 id。

        用法场景：
        - 新装一套机械臂
        - 电机 id 还没刷好

        这里要求用户一次只连接一个电机，防止总线上多个默认 id 冲突。
        """
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """读取当前 observation。

        返回内容包含两部分：
        1. 所有关节当前位置：`<motor>.pos`
        2. 所有相机最新图像：`<camera_key>`

        这是 policy / recorder 最常调用的方法之一。
        """
        # 先同步读取整只机械臂的关节位置。
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position", num_retry=10)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 再读取相机图像，并把它们拼到同一份 observation 字典里。
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """向机械臂下发一条目标关节动作。

        输入：
        - `action` 是一份以 `<motor>.pos` 为键的目标位置字典

        内部流程：
        1. 去掉 `.pos` 后缀，得到总线真正使用的 motor 名称
        2. 如果配置了 `max_relative_target`，先读当前关节位置并做安全裁剪
        3. 把目标位置同步写入 `Goal_Position`

        返回值：
        - 不是“用户原始请求的 action”
        - 而是“真正发给电机的 action”
        因为在限幅开启时，目标值可能已经被裁剪过。
        """

        # 上层统一使用 `<motor>.pos` 键名；总线写寄存器时需要纯 motor 名称。
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # 如果配置了相对动作限幅，这里先读取当前位置，再把目标位置裁到安全范围内。
        # 注意这会多一次读取，因此控制频率可能会下降。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", num_retry=10)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # 把最终目标位置同步写进所有电机。
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """断开机械臂总线与所有相机。"""
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
