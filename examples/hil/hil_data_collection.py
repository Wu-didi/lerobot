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

"""
Human-in-the-Loop (HIL) Data Collection with optional Real-Time Chunking (RTC).

Implements the RaC paradigm (https://arxiv.org/abs/2509.07953) for LeRobot. By default uses synchronous
inference (best for fast models like ACT / Diffusion Policy). Set --rtc.enabled=true for
asynchronous background inference (recommended for large models like Pi0 / Pi0.5 / SmolVLA).

The workflow:
1. Policy runs autonomously
2. Press SPACE to pause - robot holds position
3. Press 'c' to take control - human provides RECOVERY + CORRECTION
4. Press 'p' to hand control back to policy and continue recording
5. Press → to end episode (save and continue to next)
6. Reset, then do next rollout

Keyboard Controls:
    SPACE  - Pause policy (robot holds position, no recording)
    c      - Take control (start correction, recording resumes)
    p      - Resume policy after pause/correction (recording continues)
    →      - End episode (save and continue to next)
    ←      - Re-record episode
    ESC    - Stop recording and push dataset to hub

Usage:
    # 标准同步推理，适合 ACT、Diffusion Policy
    python examples/hil/hil_data_collection.py \
        --robot.type=bi_openarm_follower \
        --teleop.type=openarm_mini \
        --policy.path=path/to/pretrained_model \
        --dataset.repo_id=user/hil-dataset \
        --dataset.single_task="Fold the T-shirt properly" \
        --dataset.fps=30 \
        --interpolation_multiplier=2

    # 大模型使用 RTC，例如 Pi0、Pi0.5、SmolVLA
    python examples/hil/hil_data_collection.py \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --rtc.max_guidance_weight=5.0 \
        --rtc.prefix_attention_schedule=LINEAR \
        --robot.type=bi_openarm_follower \
        --teleop.type=openarm_mini \
        --policy.path=path/to/pretrained_model \
        --dataset.repo_id=user/hil-dataset \
        --dataset.single_task="Fold the T-shirt properly" \
        --dataset.fps=30 \
        --interpolation_multiplier=3

    # bi_openarm_follower + OpenArm Mini teleop + pi0.5 policy 的 RTC 示例
    python examples/hil/hil_data_collection.py \
        --policy.path=lerobot-data-collection/folding_final \
        --robot.type=bi_openarm_follower \
        --robot.cameras='{left_wrist: {type: opencv, index_or_path: "/dev/video4", width: 1280, height: 720, fps: 30}, base: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30}}' \
        --robot.left_arm_config.port=can0 \
        --robot.left_arm_config.side=left \
        --robot.left_arm_config.can_interface=socketcan \
        --robot.left_arm_config.disable_torque_on_disconnect=true \
        --robot.left_arm_config.max_relative_target=8.0 \
        --robot.right_arm_config.port=can1 \
        --robot.right_arm_config.side=right \
        --robot.right_arm_config.can_interface=socketcan \
        --robot.right_arm_config.disable_torque_on_disconnect=true \
        --robot.right_arm_config.max_relative_target=8.0 \
        --teleop.type=openarm_mini \
        --teleop.port_left=/dev/ttyACM1 \
        --teleop.port_right=/dev/ttyACM0 \
        --dataset.repo_id=lerobot-data-collection/hil_folding \
        --dataset.single_task="Fold the T-shirt properly" \
        --dataset.fps=30 \
        --dataset.num_episodes=50 \
        --rtc.enabled=true \
        --rtc.execution_horizon=20 \
        --rtc.max_guidance_weight=5.0 \
        --rtc.prefix_attention_schedule=LINEAR \
        --interpolation_multiplier=3 \
        --calibrate=true \
        --device=cuda
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pprint import pformat
from threading import Event, Lock, Thread
from typing import Any

import torch
from hil_utils import (
    HILDatasetConfig,
    init_keyboard_listener,
    make_identity_processors,
    print_controls,
    reset_loop,
    teleop_disable_torque,
    teleop_smooth_move_to,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import get_policy_class, make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCConfig
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
)
from lerobot.processor.relative_action_processor import to_relative_actions
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.robots.bi_koch_follower.config_bi_koch_follower import BiKochFollowerConfig  # noqa: F401
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: F401
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.bi_koch_leader.config_bi_koch_leader import BiKochLeaderConfig  # noqa: F401
from lerobot.teleoperators.openarm_mini.config_openarm_mini import OpenArmMiniConfig  # noqa: F401
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.control_utils import is_headless, predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)

# 读这个文件时，可以把它拆成三个协作组件来理解：
# 1. keyboard/pedal listener: 只修改 events 里的标志位；
# 2. rollout 主循环: 在控制频率上采观测、发动作、决定是否记录当前帧；
# 3. RTC 后台线程: 只负责提前生成 action chunk，不直接操作机器人。


# RTC 辅助函数


class ThreadSafeRobot:
    """在 RTC 主循环和后台推理线程之间串行化 robot 访问。

    RTC 模式下，主 rollout 线程和后台推理线程都可能读取机器人观测，
    而主线程还会发送动作。这个包装类只暴露 HIL 需要的少数接口，并用一把锁
    保护这些有状态的硬件调用。

    可以把它理解成“给 robot 加一层线程锁”的适配器：
    - 非 RTC 模式只有一个线程访问 robot，不需要这层包装
    - RTC 模式有多个线程共享 robot 相关访问，需要用锁避免并发访问真实硬件
    """

    def __init__(self, robot: Robot):
        """包装一个已经构造好的 robot 实例。"""
        # RTC 模式下，主 rollout 线程和后台推理线程都会读机器人观测；
        # 主线程还会发动作，所以这里用一个最小包装把访问串行化。
        self._robot = robot
        self._lock = Lock()

    def get_observation(self) -> dict[str, Any]:
        """在 RTC 共用锁保护下读取最新观测。"""
        with self._lock:
            return self._robot.get_observation()

    def send_action(self, action: dict) -> None:
        """在 RTC 共用锁保护下发送一条机器人动作。"""
        with self._lock:
            self._robot.send_action(action)

    @property
    def observation_features(self) -> dict:
        """透传底层 robot 的 observation feature 定义。"""
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        """透传底层 robot 的 action feature 定义。"""
        return self._robot.action_features

    @property
    def name(self) -> str:
        """透传底层 robot 的名称。"""
        return self._robot.name

    @property
    def robot_type(self) -> str:
        """透传 policy 输入里要用到的 robot_type 字符串。"""
        return self._robot.robot_type

    @property
    def cameras(self):
        """暴露相机配置，供 dataset/video 初始化使用。"""
        return getattr(self._robot, "cameras", {})


def _set_openarm_max_relative_target_if_missing(
    robot_cfg: RobotConfig, max_relative_target: float = 8.0
) -> None:
    """给 OpenArm follower 补一个安全的相对动作上限。

    RTC rollout 可能生成单步位移较大的 action chunk。这个函数只在用户
    没有显式配置时填入默认值，不会覆盖已有配置。
    """
    if isinstance(robot_cfg, BiOpenArmFollowerConfig):
        if robot_cfg.left_arm_config.max_relative_target is None:
            robot_cfg.left_arm_config.max_relative_target = max_relative_target
        if robot_cfg.right_arm_config.max_relative_target is None:
            robot_cfg.right_arm_config.max_relative_target = max_relative_target


def _reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_step: RelativeActionsProcessorStep | None,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    """把队列里尚未执行完的绝对动作，重新变回 policy 需要的输入空间。

    `ActionQueue` 缓存的是可以直接发给机器人的动作；但对于相对动作策略，
    下一次 RTC 推理期望拿到的是“相对于当前 state 的剩余前缀动作”。
    这个函数就负责做这次重锚定，并在需要时走同样的归一化流程。
    """
    if relative_step is None:
        return prev_actions_absolute.to(policy_device)

    # queue 里缓存的是“可直接发给机器人”的绝对动作，但相对动作策略希望拿到
    # 以当前 state 为参考系的 leftover，因此这里要反向变换一次。
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """把 RTC 前缀动作整理成固定时间长度。

    某些策略，尤其在 `torch.compile` 下，更适合接收 shape 固定的
    `prev_chunk_left_over`。因此这里会把过长的前缀截断，把过短的前缀补零。
    """
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected prev_actions to be 2D [T, A], got shape={tuple(prev_actions.shape)}")

    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]

    # torch.compile/静态图更喜欢固定 shape；不足的 prefix 用 0 填满。
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


def _resolve_action_key_order(cfg, dataset_action_names: list[str]) -> list[str]:
    """决定 policy 输出向量的各维度应该映射到哪些机器人关节键。

    policy 输出的是 tensor，而机器人最终需要的是按关节名组织的 dict。
    如果 `policy.action_feature_names` 存在并且与 dataset schema 匹配，
    就用它作为权威顺序；否则回退到 dataset 的动作顺序。
    """
    policy_action_names = getattr(cfg.policy, "action_feature_names", None)
    if not policy_action_names:
        return dataset_action_names

    # policy 输出 tensor 时只有维度顺序，没有 key 名字；这里必须决定
    # “第 i 维到底对应哪个关节”，否则后面组 robot_action dict 会串位。
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        logger.warning(
            "[RTC] policy.action_feature_names length (%d) != dataset action dim (%d); "
            "falling back to dataset order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names

    if set(dataset_action_names) != set(policy_action_names):
        logger.warning(
            "[RTC] policy.action_feature_names keys do not match dataset action keys; "
            "falling back to dataset order"
        )
        return dataset_action_names

    return policy_action_names


def _resolve_state_joint_order(
    policy_action_names: list[str] | None,
    available_joint_names: list[str],
) -> list[str]:
    """决定构造 `observation.state` 时使用的关节顺序。

    这很重要，因为很多策略默认 state 和 action 的维度顺序按关节一一对应。
    如果 policy 明确暴露了 `action_feature_names`，且它与当前硬件关节集合一致，
    HIL 就复用这个顺序，减少顺序错位风险。
    """
    if not policy_action_names:
        return available_joint_names

    # 如果 policy 明确声明了 action_feature_names，并且和当前硬件关节集合一致，
    # 就按这个顺序构造 observation.state，保证 state/action 对齐。
    policy_action_names = list(policy_action_names)
    available_set = set(available_joint_names)
    policy_set = set(policy_action_names)

    if len(policy_action_names) != len(available_joint_names) or policy_set != available_set:
        logger.warning(
            "policy.action_feature_names does not match available state joints; "
            "falling back to robot observation order"
        )
        return available_joint_names

    logger.info("Using policy.action_feature_names order for observation.state mapping")
    return policy_action_names


def _start_pedal_listener(events: dict):
    """如果系统支持 evdev，就启动脚踏板监听线程。

    脚踏板只负责 HIL 的控制权切换：
    policy -> pause -> takeover -> resume policy。
    episode 的保存/推进仍然由键盘右箭头负责。

    和键盘监听器一样，这个线程只修改 `events`，真正控制机器人和写数据的
    仍然是 rollout 主循环。
    """
    import threading

    try:
        from evdev import InputDevice, categorize, ecodes
    except ImportError:
        logging.warning("[Pedal] evdev not installed - pedal support disabled")
        return

    pedal_device = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    key_left = "KEY_A"
    key_right = "KEY_C"
    # 这个脚踏板被系统识别成键盘设备，不同踏板映射到不同 keycode。

    # pedal_reader 持续消费 evdev 事件，并翻译成与键盘相同的状态机信号。
    def pedal_reader():
        try:
            dev = InputDevice(pedal_device)
            logger.info(f"[Pedal] Connected: {dev.name}")

            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue

                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]

                if key.keystate != 1:
                    continue

                if events["in_reset"]:
                    if code in [key_left, key_right]:
                        events["start_next_episode"] = True
                else:
                    if code not in [key_left, key_right]:
                        continue

                    if events["correction_active"]:
                        # 脚踏板再次触发时，把控制权交回 policy。
                        events["resume_policy"] = True
                    elif events["policy_paused"]:
                        # 暂停后第一次触发，等价于键盘里的“开始人工接管”。
                        events["start_next_episode"] = True
                    else:
                        # 正常 autonomous rollout 中触发，先进入 pause。
                        events["policy_paused"] = True

        except FileNotFoundError:
            logging.info(f"[Pedal] Device not found: {pedal_device}")
        except PermissionError:
            logging.warning(f"[Pedal] Permission denied for {pedal_device}")
        except Exception as e:
            logging.warning(f"[Pedal] Error: {e}")

    thread = threading.Thread(target=pedal_reader, daemon=True)
    thread.start()


def _rtc_inference_thread(
    policy: PreTrainedPolicy,
    obs_holder: dict,
    obs_lock: Lock,
    hw_features: dict,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    queue_holder: dict,
    shutdown_event: Event,
    policy_active: Event,
    compile_warmup_done: Event,
    cfg,
):
    """在后台异步生成 RTC action chunk。

    这个线程的职责是：
    1. 等待主 rollout 循环把 `policy_active` 置为真。
    2. 从 `obs_holder` 读取最新观测快照。
    3. 执行 preprocess -> `predict_action_chunk` -> postprocess。
    4. 把生成好的 chunk 合并进 `ActionQueue`，供主 rollout 循环稍后消费。

    它不会直接给机器人发动作，也不会直接写 dataset frame；它只负责准备未来动作。
    """
    latency_tracker = LatencyTracker()
    time_per_chunk = 1.0 / cfg.dataset.fps
    # 队列不能太长，否则新 chunk 很难替换过期规划；也不能太短，
    # 否则机器人可能没有可执行动作。
    threshold = 30
    policy_device = policy.config.device
    stats_window_start = time.perf_counter()
    policy_inference_count = 0
    latency_sum_s = 0.0
    inference_count = 0
    warmup_required = max(1, int(cfg.compile_warmup_inferences)) if cfg.use_torch_compile else 0

    relative_step = next(
        (
            step
            for step in preprocessor.steps
            if isinstance(step, RelativeActionsProcessorStep) and step.enabled
        ),
        None,
    )
    normalizer_step = next(
        (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
        None,
    )
    if relative_step is not None:
        # RTC prefix 来自上一段 chunk 还没执行完的动作。若策略使用相对动作，
        # 需要围绕当前 state 重新锚定这些 leftover。
        if relative_step.action_names is None:
            cfg_action_names = getattr(cfg.policy, "action_feature_names", None)
            if cfg_action_names:
                relative_step.action_names = list(cfg_action_names)
            else:
                fallback_action_names = obs_holder.get("action_feature_names")
                if fallback_action_names:
                    relative_step.action_names = list(fallback_action_names)
        logger.info("[RTC] Relative actions enabled: re-anchoring RTC prefix to current state")

    while not shutdown_event.is_set():
        # 这个线程只在 policy_active=True 时工作。主 rollout 在 pause/reset/
        # human takeover 阶段会 clear 该标志，让后台停止生成过期 chunk。
        if not policy_active.is_set():
            # rollout 循环会在暂停、人工接管、等待下一条 episode 时清掉该标志。
            time.sleep(0.01)
            continue

        queue = queue_holder.get("queue")
        with obs_lock:
            obs = obs_holder.get("obs")
        if queue is None or obs is None:
            time.sleep(0.01)
            continue

        if queue.qsize() <= threshold:
            try:
                # 当主循环快把动作队列消费完时，生成新的 action chunk。
                current_time = time.perf_counter()
                idx_before = queue.get_action_index()
                prev_actions = queue.get_left_over()

                # RTC 会估计推理耗时，并跳过推理期间已经流逝的控制步。
                latency = latency_tracker.max()
                delay = math.ceil(latency / time_per_chunk) if latency else 0

                # 把最新机器人观测转换成 policy 可直接处理的 batch。
                obs_batch = build_dataset_frame(hw_features, obs, prefix="observation")
                for name in obs_batch:
                    obs_batch[name] = torch.from_numpy(obs_batch[name])
                    if "image" in name:
                        obs_batch[name] = obs_batch[name].float() / 255
                        obs_batch[name] = obs_batch[name].permute(2, 0, 1).contiguous()
                    obs_batch[name] = obs_batch[name].unsqueeze(0).to(policy_device)

                obs_batch["task"] = [cfg.dataset.single_task]
                obs_batch["robot_type"] = obs_holder.get("robot_type", "unknown")

                preprocessed = preprocessor(obs_batch)

                if prev_actions is not None and relative_step is not None and OBS_STATE in obs_batch:
                    # 队列里的 leftover 是可直接执行的机器人动作；作为 RTC prefix
                    # 输入模型前，需要先重新转换到模型动作空间。
                    prev_actions_absolute = queue.get_processed_left_over()
                    if prev_actions_absolute is not None and prev_actions_absolute.numel() > 0:
                        prev_actions = _reanchor_relative_rtc_prefix(
                            prev_actions_absolute=prev_actions_absolute,
                            current_state=obs_batch[OBS_STATE],
                            relative_step=relative_step,
                            normalizer_step=normalizer_step,
                            policy_device=policy_device,
                        )

                if prev_actions is not None:
                    prev_actions = _normalize_prev_actions_length(
                        prev_actions, target_steps=cfg.rtc.execution_horizon
                    )

                actions = policy.predict_action_chunk(
                    preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
                )

                original = actions.squeeze(0).clone()
                processed = postprocessor(actions).squeeze(0)
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                inference_count += 1
                is_warmup_inference = cfg.use_torch_compile and inference_count <= warmup_required
                if is_warmup_inference:
                    latency_tracker.reset()
                else:
                    latency_tracker.add(new_latency)
                # original actions 留给下一次 RTC prefix 使用；processed actions
                # 是主 rollout 循环真正发给机器人的动作。
                queue.merge(original, processed, new_delay, idx_before)
                policy_inference_count += 1
                latency_sum_s += new_latency
                if (
                    is_warmup_inference
                    and inference_count >= warmup_required
                    and not compile_warmup_done.is_set()
                ):
                    compile_warmup_done.set()
                    logger.info(
                        "[RTC] Compile warmup complete (%d/%d inferences)",
                        inference_count,
                        warmup_required,
                    )
                logger.debug("[RTC] Inference latency=%.2fs, queue=%d", new_latency, queue.qsize())
            except Exception as e:
                logger.error("[RTC] Error: %s", e)
                time.sleep(0.5)
        else:
            time.sleep(0.01)

        now = time.perf_counter()
        if cfg.log_hz and (window_elapsed := now - stats_window_start) >= cfg.hz_log_interval_s:
            policy_hz = policy_inference_count / window_elapsed
            avg_latency_ms = (
                (latency_sum_s / policy_inference_count * 1000.0) if policy_inference_count else 0.0
            )
            logger.info(
                "[HIL RTC rates] policy=%.1f Hz | avg_inference=%.1f ms | queue=%d",
                policy_hz,
                avg_latency_ms,
                queue.qsize(),
            )
            stats_window_start = now
            policy_inference_count = 0
            latency_sum_s = 0.0


# 配置


@dataclass
class HILConfig:
    """HIL 数据采集脚本的顶层 CLI 配置。

    这份配置大致分成四组：
    1. `robot` / `teleop`：硬件如何构造和连接。
    2. `dataset`：数据如何记录和存储。
    3. `policy`：使用哪个预训练策略。
    4. `rtc` 与插值参数：策略输出如何在时间上调度执行。
    """

    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: HILDatasetConfig
    policy: PreTrainedConfig | None = None
    rtc: RTCConfig = field(default_factory=RTCConfig)
    # interpolation_multiplier 控制“机器人执行频率 / dataset 采样频率”。
    # 例如 fps=30, multiplier=3 时，机器人以 90Hz 执行动作，但默认仍只录 30Hz。
    interpolation_multiplier: int = 2
    record_interpolated_actions: bool = False
    display_data: bool = True
    play_sounds: bool = True
    resume: bool = False
    device: str = "cuda"
    use_torch_compile: bool = False
    compile_warmup_inferences: int = 2
    calibrate: bool = False
    log_hz: bool = True
    hz_log_interval_s: float = 2.0

    def __post_init__(self):
        """在 CLI 和路径解析完成后，补加载预训练 policy 配置。"""
        # policy.path 是 path-like 字段，所以等 draccus 解析完路径和
        # policy 下的 CLI override 后，再加载 pretrained config。
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("policy.path is required")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """告诉 parser 哪个嵌套字段应按 path 处理。"""
        return ["policy"]


# Rollout 循环


@safe_stop_image_writer
def _rollout_sync(
    robot: Robot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    events: dict,
    cfg: HILConfig,
):
    """执行一条“同步推理模式”的 HIL episode。

    这是最容易理解的版本：
    1. 从机器人读取 observation。
    2. 如果当前由 policy 控制，就在这个主循环里直接同步推理。
    3. 如果当前处于人工纠正阶段，就转而执行 teleop 动作。
    4. 把最终选定的动作发给机器人。
    5. 根据记录策略，决定是否把当前 observation/action 写入 episode buffer。

    这个函数只负责填充 `dataset` 当前的内存 episode buffer，不负责保存或丢弃。

    可以把这个函数理解成一个固定频率运行的控制循环。每个 tick 都做同样 5 件事：
    1. 先处理键盘事件带来的状态切换。
    2. 读取当前机器人观测，并整理成 obs_frame。
    3. 决定“这一拍动作由谁提供”：
       - 人类接管时，动作用 teleop.get_action()
       - 暂停但未接管时，重复发送 last_action 让机器人保持姿态
       - 正常 autonomous 模式时，动作用 policy.predict_action()
    4. 必要时把当前 observation/action 写入当前 episode buffer。
    5. sleep 到下一个控制周期。

    这个同步版没有后台线程，也没有 action queue，所有策略推理都在主循环里直接完成，
    所以它最适合用来理解 HIL 的基本控制逻辑。
    """
    fps = cfg.dataset.fps
    device = get_safe_torch_device(cfg.device)
    stream_online = bool(cfg.dataset.streaming_encoding)
    # 默认只按 dataset fps 记录一帧；插值产生的中间控制步通常不写入数据集，
    # 除非显式打开 record_interpolated_actions。
    record_stride = 1 if cfg.record_interpolated_actions else max(1, cfg.interpolation_multiplier)

    # episode 边界处重置 policy 和 processor，避免循环状态或动作队列
    # 泄漏到下一条 rollout。
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    frame_buffer: list[dict] = []
    teleop_disable_torque(teleop)

    # 下面几个变量是理解这个循环的关键：
    # was_paused:
    #   防止“按一次暂停键”对应的初始化逻辑被重复执行多次。
    # waiting_for_takeover:
    #   已经暂停，但人类还没正式开始接管；此时机器人只保持原位。
    # last_action:
    #   最近一次真正发给机器人的动作。暂停时会重复发送它来“定住”机器人。
    # robot_action:
    #   当前这一拍最终决定要发给机器人的动作；也会作为记录时的 action 来源。
    was_paused = False
    waiting_for_takeover = False
    last_action: dict[str, Any] | None = None
    robot_action: dict[str, Any] = {}
    action_keys = list(dataset.features[ACTION]["names"])
    obs_state_names = list(dataset.features[f"{OBS_STR}.state"]["names"])
    obs_image_names = [
        key.removeprefix(f"{OBS_STR}.images.")
        for key in dataset.features
        if key.startswith(f"{OBS_STR}.images.")
    ]

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(fps)
    # 于是这个 rollout 有两套频率：
    # - policy/recording 频率约为 fps
    # - robot command 频率约为 fps * interpolation_multiplier
    #
    # 直觉上可以这样理解：
    # policy 不一定每个控制 tick 都重新推理一次，而是先给出较稀疏的动作点；
    # interpolator 再把这些动作点插值成更高频的机器人控制命令。

    timestamp = 0.0
    record_tick = 0
    start_t = time.perf_counter()
    stats_window_start = start_t
    policy_inference_count = 0
    robot_command_count = 0

    while timestamp < cfg.dataset.episode_time_s:
        loop_start = time.perf_counter()

        # 这个 while 就是一条 episode 的主控制循环。
        #
        # 可以把状态分成 4 种：
        # 1. 正常 autonomous 模式
        #    policy 产生命令，机器人执行，数据被记录。
        # 2. paused 模式
        #    人按下 SPACE 后，policy 暂停；机器人重复 last_action 保持姿态。
        # 3. waiting_for_takeover 模式
        #    已经暂停，系统正在等待人按 c 正式开始接管。
        # 4. correction_active 模式
        #    人类动作直接控制机器人，这些动作也会被记录成训练数据。
        #
        # 每一轮循环开头先根据 events 决定是否发生状态切换。
        if events["exit_early"]:
            # 退出键会结束当前 rollout。右箭头保存当前 partial episode；
            # 左箭头标记为重录；ESC 会在外层停止整个采集。
            events["exit_early"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            events["resume_policy"] = False
            break

        if events["resume_policy"] and (
            events["policy_paused"] or events["correction_active"] or waiting_for_takeover
        ):
            # 用户按 p 把控制权交还给 policy。
            #
            # 为什么这里要 reset 这么多东西？
            # 因为人工接管期间，policy 的历史上下文、插值器里的旧动作都已经“过时”了。
            # 如果不清空，恢复后 policy 可能会沿着旧状态继续规划，导致动作跳变。
            events["resume_policy"] = False
            events["start_next_episode"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            waiting_for_takeover = False
            was_paused = False
            last_action = None
            interpolator.reset()
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        if events["policy_paused"] and not was_paused:
            # 第一次进入 pause 状态时，只做一次“暂停初始化”：
            # 1. 读取当前机器人姿态
            # 2. 如有需要，可把 teleop 对齐到该姿态
            # 3. 进入 waiting_for_takeover，开始等待 c / p
            #
            # 注意：这里只是“暂停”，还不是“人工接管”。
            obs = robot.get_observation()
            robot_pos = {
                k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
            }
            # teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)
            events["start_next_episode"] = False
            waiting_for_takeover = True
            was_paused = True
            interpolator.reset()

        if waiting_for_takeover and events["start_next_episode"]:
            # 在 rollout 阶段，start_next_episode 这个标志被复用成“开始接管”。
            # 也就是说：
            # SPACE 只是暂停
            # c 才会把 waiting_for_takeover -> correction_active
            teleop_disable_torque(teleop)
            events["start_next_episode"] = False
            events["correction_active"] = True
            waiting_for_takeover = False

        # 无论当前是谁控制机器人，每一轮都会先读取一份最新观测。
        # 这份观测有两个用途：
        # 1. 给 policy 推理作为输入
        # 2. 给 dataset 记录作为当前帧 observation
        obs = robot.get_observation()
        obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
        obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
        # build_dataset_frame:
        # 输入: dataset.features 定义、原始观测字典 obs_filtered，以及前缀 "observation"
        # 输出: 一个字段名完全对齐 dataset schema 的 obs_frame 字典
        # 作用: 把“机器人当前观测”整理成后续 policy 推理和数据记录都能复用的格式。
        obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)

        if events["correction_active"]:
            # 分支 1: 人工接管中。
            #
            # 此时“人类动作”就是机器人真正执行的动作，同时也是训练标签。
            # 所以这一分支里不会跑 policy，而是直接：
            # teleop -> robot -> dataset
            # teleop.get_action:
            # 输入: 无，直接读取当前 teleop 硬件状态
            # 输出: 一个按关节名组织的动作字典
            # 作用: 代表“此刻人类希望机器人执行的动作”，在人工接管阶段它会
            # 直接成为训练标签写入数据集。
            robot_action = teleop.get_action()
            # robot.send_action:
            # 输入: 一个机器人动作字典 robot_action
            # 输出: 按接口约定通常会返回“实际发送出去的动作”，但在 HIL 里这里
            # 主要关心它的副作用: 通过底层驱动把目标动作发给真实机械臂。
            robot.send_action(robot_action)
            robot_command_count += 1
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            if record_tick % record_stride == 0:
                frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                if stream_online:
                    # dataset.add_frame:
                    # 输入: 一帧完整数据 frame，通常包含 observation/action/task
                    # 输出: 无；副作用是把该帧追加到当前 episode buffer
                    # 作用: 先积累帧，稍后由 save_episode() 一次性提交成一条 episode。
                    dataset.add_frame(frame)
                else:
                    # 非 streaming 模式下，先把帧缓存在内存里，episode 结束后再统一 add_frame。
                    frame_buffer.append(frame)
            record_tick += 1

        elif waiting_for_takeover or events["policy_paused"]:
            # 分支 2: 已暂停，但人类还没开始接管，或者仍停留在 pause 状态。
            #
            # 这里的策略是“保持不动”：
            # 不再请求 policy 新动作，也不读取 teleop 动作，而是重复发送 last_action。
            # 这能让机器人维持在暂停前的目标姿态附近。
            if last_action:
                robot.send_action(last_action)
                robot_command_count += 1

        else:
            # 分支 3: 正常 autonomous 模式。
            #
            # 这里只有在插值器“需要新的基础动作点”时才会真正跑一次 policy；
            # 其余控制 tick 会直接消费插值器输出，减少推理频率。
            if interpolator.needs_new_action():
                # 同步模式会在这里阻塞等待 policy 返回下一条动作，
                # 因此更适合推理较快的策略。
                # predict_action:
                # 输入: 当前 observation（obs_frame）、policy、设备、pre/post processor、
                # task 和 robot_type
                # 输出: policy 输出的一条动作张量，语义上还处在“模型动作空间”
                # 作用: 完成一次完整的 推理前处理 -> 模型前向 -> 推理后处理。
                action_values = predict_action(
                    observation=obs_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.dataset.single_task,
                    robot_type=robot.robot_type,
                )
                policy_inference_count += 1
                # make_robot_action:
                # 输入: 模型输出张量 action_values，以及 dataset.features 中记录的
                # action 名称顺序
                # 输出: 一个按关节名组织的动作字典，例如 {"left_joint_1.pos": ...}
                # 作用: 把“按维度排列的 tensor”变成“机器人接口真正需要的 dict”。
                robot_action = make_robot_action(action_values, dataset.features)
                # interpolator 处理的是“有顺序的向量”，不是 dict，所以这里要
                # 按 action_keys 顺序把动作字典重新打平成 tensor。
                action_tensor = torch.tensor([robot_action[k] for k in action_keys])
                interpolator.add(action_tensor)

            interp_action = interpolator.get()
            if interp_action is not None:
                # interpolator.get() 取到的是当前控制 tick 要执行的那一小步动作。
                # 它可能是 policy 原始动作，也可能是相邻动作点之间的插值结果。
                robot_action = {k: interp_action[i].item() for i, k in enumerate(action_keys)}
                robot.send_action(robot_action)
                robot_command_count += 1
                last_action = robot_action
                action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
                if record_tick % record_stride == 0:
                    # 默认只按 cfg.dataset.fps 记录数据；即使机器人通过插值
                    # 以更高频率控制，也不会每个控制步都写入数据集。
                    frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                    if stream_online:
                        dataset.add_frame(frame)
                    else:
                        frame_buffer.append(frame)
                record_tick += 1

        # 可视化只用于调试/观测，不参与控制决策。
        if cfg.display_data and robot_action:
            log_rerun_data(observation=obs_filtered, action=robot_action)

        # 把当前循环补齐到固定控制周期，尽量维持稳定的控制频率。
        dt = time.perf_counter() - loop_start
        if (sleep_time := control_interval - dt) > 0:
            precise_sleep(sleep_time)
        now = time.perf_counter()
        timestamp = now - start_t

        if cfg.log_hz and (window_elapsed := now - stats_window_start) >= cfg.hz_log_interval_s:
            policy_hz = policy_inference_count / window_elapsed
            robot_hz = robot_command_count / window_elapsed
            logger.info(
                "[HIL rates] policy=%.1f Hz (target=%.1f) | robot=%.1f Hz (target=%.1f)",
                policy_hz,
                fps,
                robot_hz,
                fps * cfg.interpolation_multiplier,
            )
            stats_window_start = now
            policy_inference_count = 0
            robot_command_count = 0

    # 离开 episode 前，确保 teleop 处于可手动操作的状态。
    teleop_disable_torque(teleop)

    if not stream_online:
        # 非 streaming 模式下，真正写入 dataset buffer 的时机在 episode 末尾。
        for frame in frame_buffer:
            dataset.add_frame(frame)

# @safe_stop_image_writer:
# 这是一个“异常清理型装饰器”。
# 它不会改变 _rollout_rtc 的业务逻辑；它做的事情是：
# 如果函数内部抛异常，并且 dataset 里挂着 image writer，
# 就先尝试把 image writer 安全停掉，再把异常继续往外抛。
# 这样可以减少“主逻辑已经报错退出，但后台图像写线程/进程还挂着”的情况。
@safe_stop_image_writer
def _rollout_rtc(
    robot,
    teleop: Teleoperator,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset: LeRobotDataset,
    events: dict,
    cfg: HILConfig,
    queue_holder: dict,
    obs_holder: dict,
    obs_lock: Lock,
    policy_active: Event,
    compile_warmup_done: Event,
    hw_features: dict,
):
    """执行一条“RTC 异步推理模式”的 HIL episode。

    与 `_rollout_sync` 不同，这个函数不会在主循环里阻塞等待 policy 推理，
    而是：
    1. 把最新 observation 发布到 `obs_holder`。
    2. 让 `_rtc_inference_thread` 在后台持续填充 `ActionQueue`。
    3. 从队列里取基础动作，经过插值后发送给机器人。
    4. 处理与同步模式相同的 pause / takeover / resume / rerecord 状态切换。

    和同步版一样，它也只负责向当前 episode buffer 追加 frame。

    可以把它理解成“同步版 + 后台动作生产线程”：
    1. 主循环仍然负责状态切换、发机器人动作、记录数据。
    2. 但 policy 推理不再阻塞主循环，而是由 `_rtc_inference_thread`
       提前把未来一段动作放进 `ActionQueue`。
    3. 主循环只需要按控制节拍从 `ActionQueue` 里取基础动作，再交给
       `ActionInterpolator` 插值成更高频的机器人动作。

    所以 RTC 版的关键心智模型是：
    - 后台线程负责“生产动作”
    - 主循环负责“消费动作并执行”
    - `obs_holder` / `queue_holder` 是两者之间的交接点
    """
    fps = cfg.dataset.fps
    stream_online = bool(cfg.dataset.streaming_encoding)
    record_stride = 1 if cfg.record_interpolated_actions else max(1, cfg.interpolation_multiplier)

    # RTC rollout 不在主循环里直接跑 policy，而是消费后台推理线程
    # 预先放入队列的 action chunk。
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    frame_buffer: list[dict] = []
    teleop_disable_torque(teleop)

    # 这些变量和同步版很像，但 RTC 还多了“队列”和“观测发布节拍”的概念：
    # was_paused / waiting_for_takeover / last_action:
    #   语义和同步版相同，用于管理 pause -> takeover -> resume 的状态流转。
    # dataset_action_keys:
    #   dataset schema 里动作字段的顺序。
    # action_keys:
    #   真正用于把 action tensor 还原成 dict 的顺序；RTC 下优先使用
    #   policy.action_feature_names，避免动作维度和关节名错位。
    was_paused = False
    waiting_for_takeover = False
    last_action: dict[str, Any] | None = None
    dataset_action_keys = list(dataset.features[ACTION]["names"])
    action_keys = _resolve_action_key_order(cfg, dataset_action_keys)
    if action_keys != dataset_action_keys:
        logger.info("[RTC] Using policy.action_feature_names order for action tensor mapping")
    else:
        logger.info("[RTC] Using dataset action feature order for action tensor mapping")
    obs_state_names = list(dataset.features[f"{OBS_STR}.state"]["names"])
    obs_image_names = [
        key.removeprefix(f"{OBS_STR}.images.")
        for key in dataset.features
        if key.startswith(f"{OBS_STR}.images.")
    ]

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(fps)
    # RTC 模式依然用插值器提高执行频率；区别只是“基础动作”不再由主线程同步推理，
    # 而是从后台线程维护的 ActionQueue 里取。
    #
    # 所以这里依然存在两层频率：
    # - 后台 policy/queue 的基础动作频率，大约是 dataset fps
    # - 主循环真正发给机器人的控制频率，是 fps * interpolation_multiplier

    robot_action: dict[str, Any] = {}
    timestamp = 0.0
    start_t = time.perf_counter()
    stats_window_start = start_t
    robot_command_count = 0
    record_tick = 0
    # RTC 下不会每个控制 tick 都重新拉机器人观测给 policy；
    # 正常 autonomous 情况下，按 dataset fps 发布观测就够了。
    # 但在 pause/correction 阶段，为了及时反映人工操作，会更频繁取观测。
    obs_poll_interval = 1.0 / fps
    last_obs_poll_t = 0.0
    obs_filtered: dict[str, Any] = {}
    obs_frame: dict[str, Any] = {}
    warmup_wait_logged = False
    warmup_queue_flushed = False

    while timestamp < cfg.dataset.episode_time_s:
        loop_start = time.perf_counter()

        # RTC 版的状态机和同步版基本一致：
        # 1. 正常 autonomous：后台线程负责产动作，主线程消费并执行
        # 2. paused：停止后台推理，机器人重复 last_action 保持姿态
        # 3. waiting_for_takeover：已暂停，等待人正式开始接管
        # 4. correction_active：人类动作直接控制机器人，并被记录进数据集
        #
        # 不同点在于：这里还要额外处理 queue、obs_holder、policy_active
        # 这几个 RTC 共享状态。
        if events["exit_early"]:
            # 退出键会结束当前 rollout。右箭头保存当前 partial episode；
            # 左箭头标记为重录；ESC 会在外层停止整个采集。
            events["exit_early"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            events["resume_policy"] = False
            break

        if events["resume_policy"] and (
            events["policy_paused"] or events["correction_active"] or waiting_for_takeover
        ):
            # 人工阶段结束，控制权交回 policy。
            #
            # RTC 比同步版多做的一步是“清空动作队列”：
            # 因为队列里的动作是基于人工介入前的旧观测生成的，继续执行它们会很危险。
            events["resume_policy"] = False
            events["start_next_episode"] = False
            events["policy_paused"] = False
            events["correction_active"] = False
            waiting_for_takeover = False
            was_paused = False
            last_action = None
            interpolator.reset()
            # 人工干预后丢弃已排队动作，因为它们是基于旧观测规划出来的。
            queue_holder["queue"] = ActionQueue(cfg.rtc)
            policy_active.clear()
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        if events["policy_paused"] and not was_paused:
            # 第一次进入 pause 时：
            # 1. 先 clear policy_active，告诉后台线程“先不要再产新动作了”
            # 2. 读取当前机器人姿态；必要时可把 teleop 对齐过来
            # 3. 进入 waiting_for_takeover，等待 c / p
            policy_active.clear()
            obs = robot.get_observation()
            robot_pos = {
                k: v for k, v in obs.items() if k.endswith(".pos") and k in robot.observation_features
            }
            # teleop_smooth_move_to(teleop, robot_pos, duration_s=2.0, fps=50)
            events["start_next_episode"] = False
            waiting_for_takeover = True
            was_paused = True
            interpolator.reset()

        if waiting_for_takeover and events["start_next_episode"]:
            # 在 RTC rollout 里，start_next_episode 这里同样表示“开始人工接管”。
            # 进入 correction_active 时，同时把 queue 也换成新的空队列，
            # 避免旧 chunk 遗留到人工阶段之后。
            teleop_disable_torque(teleop)
            events["start_next_episode"] = False
            events["correction_active"] = True
            waiting_for_takeover = False
            queue_holder["queue"] = ActionQueue(cfg.rtc)

        now_for_obs = time.perf_counter()
        # 正常情况下按 dataset fps 采样观测用于 policy 输入和记录；
        # 纠正/暂停时每个控制 tick 都采样，以获得更新的人工反馈。
        should_poll_obs = (
            not obs_filtered
            or (now_for_obs - last_obs_poll_t) >= obs_poll_interval
            or events["correction_active"]
            or waiting_for_takeover
            or events["policy_paused"]
        )
        if should_poll_obs:
            # 这里做两件事：
            # 1. 构造 obs_frame，供当前 tick 记录数据使用
            # 2. 把最新 obs_filtered 发布到 obs_holder，供后台 RTC 推理线程读取
            obs = robot.get_observation()
            obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
            obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
            obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)
            with obs_lock:
                obs_holder["obs"] = obs_filtered
            last_obs_poll_t = now_for_obs

        if events["correction_active"]:
            # 分支 1: 人工接管中。
            # 和同步版一样，teleop 动作会直接控制机器人，也会作为训练标签写入数据集。
            robot_action = teleop.get_action()
            robot.send_action(robot_action)
            robot_command_count += 1
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            if record_tick % record_stride == 0:
                frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                if stream_online:
                    dataset.add_frame(frame)
                else:
                    frame_buffer.append(frame)
            record_tick += 1

        elif waiting_for_takeover or events["policy_paused"]:
            # 分支 2: 暂停但未接管。
            # 由于此时后台 policy 线程被停掉了，也不会产生新动作，
            # 所以只能重复发 last_action，让机器人尽量维持当前位置。
            if last_action:
                robot.send_action(last_action)
                robot_command_count += 1

        else:
            # 分支 3: 正常 RTC autonomous 模式。
            #
            # 这一分支的关键是“主线程不直接推理”：
            # - 先确保后台线程被允许工作(policy_active.set())
            # - 后台线程不断往 queue 里填 action chunk
            # - 主线程这里只负责从 queue 取动作并执行
            if not policy_active.is_set():
                # 允许后台推理线程开始填充 action queue。
                policy_active.set()

            if cfg.use_torch_compile and not compile_warmup_done.is_set():
                if not warmup_wait_logged:
                    logger.info(
                        "[RTC] Waiting for compile warmup (%d inferences) before policy rollout",
                        max(1, int(cfg.compile_warmup_inferences)),
                    )
                    warmup_wait_logged = True
            else:
                if cfg.use_torch_compile and not warmup_queue_flushed:
                    # 丢弃 warmup 阶段生成的动作；这些动作只用于触发编译
                    # 和获得稳定的首次延迟估计。
                    queue_holder["queue"] = ActionQueue(cfg.rtc)
                    interpolator.reset()
                    warmup_queue_flushed = True
                    logger.info("[RTC] Warmup queue cleared; starting live policy rollout")

                queue = queue_holder["queue"]

                if interpolator.needs_new_action():
                    # RTC 将 policy 频率和机器人控制频率解耦：queue.get()
                    # 提供基础 fps 动作，再由插值器提升执行频率。
                    #
                    # 这里的 new_action 不是“刚算出来的动作”，而是后台线程更早一点
                    # 放进队列里的动作。因此主线程即使不跑模型，也能持续控制机器人。
                    new_action = queue.get() if queue else None
                    if new_action is not None:
                        interpolator.add(new_action.cpu())

                action_tensor = interpolator.get()
                if action_tensor is not None:
                    # 队列和插值器里处理的是 tensor；真正发给 robot 前要先按 action_keys
                    # 还原成带关节名的 dict。
                    robot_action = {
                        k: action_tensor[i].item()
                        for i, k in enumerate(action_keys)
                        if i < len(action_tensor)
                    }
                    robot.send_action(robot_action)
                    robot_command_count += 1
                    last_action = robot_action
                    action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
                    if record_tick % record_stride == 0:
                        # 默认让记录样本对齐 dataset fps；只有开启
                        # record_interpolated_actions 时才记录每个控制 tick。
                        frame = {**obs_frame, **action_frame, "task": cfg.dataset.single_task}
                        if stream_online:
                            dataset.add_frame(frame)
                        else:
                            frame_buffer.append(frame)
                    record_tick += 1

        # RTC 和同步版一样，循环尾部负责补足固定控制节拍。
        dt = time.perf_counter() - loop_start
        if (sleep_time := control_interval - dt) > 0:
            precise_sleep(sleep_time)
        now = time.perf_counter()
        timestamp = now - start_t

        if cfg.log_hz and (window_elapsed := now - stats_window_start) >= cfg.hz_log_interval_s:
            robot_hz = robot_command_count / window_elapsed
            logger.info(
                "[HIL RTC rates] robot=%.1f Hz (target=%.1f)",
                robot_hz,
                fps * cfg.interpolation_multiplier,
            )
            stats_window_start = now
            robot_command_count = 0

    # episode 结束时，明确告诉后台线程停止为这条 rollout 继续产动作。
    policy_active.clear()
    teleop_disable_torque(teleop)

    if not stream_online:
        # 非 streaming 模式下，episode 期间先缓存 frame，最后统一写入 dataset buffer。
        for frame in frame_buffer:
            dataset.add_frame(frame)


# 主采集函数

# @parser.wrap():
# 这是一个“CLI 配置注入型装饰器”。
# 它的作用是：当你从命令行执行脚本时，不需要自己手写
# `cfg = parser.parse(...)` 之类的代码；装饰器会在调用 `hil_collect()` 前，
# 自动完成：
# 1. 解析命令行参数
# 2. 构造 HILConfig 实例
# 3. 把这个 cfg 作为第一个参数传给 hil_collect
#
# 所以虽然函数签名写的是 `hil_collect(cfg: HILConfig)`，
# 但在 main() 里可以直接无参调用 `hil_collect()`。
@parser.wrap()
def hil_collect(cfg: HILConfig) -> LeRobotDataset:
    """统筹整个 HIL 数据采集会话。

    这是命令行真正调用的顶层入口，负责：
    1. 初始化日志、dataset schema、policy、robot 和 teleop。
    2. 按需启动 RTC 辅助线程和监听器。
    3. 按 episode 循环调用对应的 rollout 函数。
    4. 处理保存、重录、reset、收尾清理和可选的 push-to-hub。

    如果你想从顶层理解整个脚本，应该先从这个函数开始读。
    """
    init_logging()
    logger.info(pformat(cfg.__dict__))

    use_rtc = cfg.rtc.enabled

    if use_rtc:
        # OpenArm follower 在 RTC relative action 下通常需要安全的相对位移上限，
        # 否则一次 chunk 里的动作变化可能过大。
        _set_openarm_max_relative_target_if_missing(cfg.robot, max_relative_target=8.0)

    if cfg.display_data:
        init_rerun(session_name="hil_collection")

    # make_robot_from_config:
    # 输入: cfg.robot，一个已经由 CLI 解析完成的 RobotConfig 子类实例
    # 输出: 一个“尚未 connect”的 Robot 对象。
    # 作用: 把配置转换成统一的机器人运行时接口。后面的 HIL 代码不会关心底层
    # 是哪种机械臂，只通过这个对象的 get_observation / send_action /
    # observation_features / action_features 等接口来交互。
    robot_raw = make_robot_from_config(cfg.robot)
    # make_teleoperator_from_config:
    # 输入: cfg.teleop，一个 TeleoperatorConfig 子类实例
    # 输出: 一个“尚未 connect”的 Teleoperator 对象。
    # 作用: 把人类操作者的输入设备包装成统一接口。后面通过它读取人工动作
    # (teleop.get_action)，并在支持时执行 enable_torque / disable_torque /
    # write_goal_positions 这类 leader 设备控制。
    teleop = make_teleoperator_from_config(cfg.teleop)

    # make_identity_processors:
    # 输入: 无
    # 输出: (teleop_proc, obs_proc) 两个 identity processor pipeline
    # 作用: 它们本身不改写 action/observation，只是借用 processor framework
    # 来生成标准的 dataset feature schema，方便后面构建 LeRobotDataset。
    teleop_proc, obs_proc = make_identity_processors()

    # dataset feature 名称来自机器人硬件；如果 policy 提供了动作顺序，
    # state 向量顺序也可以和 policy 的 action 顺序对齐。
    action_features_hw = {k: v for k, v in robot_raw.action_features.items() if k.endswith(".pos")}
    all_observation_features = robot_raw.observation_features
    available_joint_names = [
        key for key, value in all_observation_features.items() if key.endswith(".pos") and value is float
    ]
    ordered_joint_names = _resolve_state_joint_order(
        getattr(cfg.policy, "action_feature_names", None),
        available_joint_names,
    )
    observation_features_hw = {
        joint_name: all_observation_features[joint_name] for joint_name in ordered_joint_names
    }
    for key, value in all_observation_features.items():
        if isinstance(value, tuple):
            observation_features_hw[key] = value

    # combine_feature_dicts + aggregate_pipeline_dataset_features:
    # 输入: 机器人硬件侧的原始 action/observation feature 定义，以及上面两个
    # identity processor pipeline。
    # 输出: dataset_features，一个完整的 LeRobotDataset schema，描述每一帧里
    # 会保存哪些键、每个键的类型/形状是什么。
    # 作用: 先把“硬件原始字段”转成“数据集字段定义”，后面的 build_dataset_frame /
    # dataset.add_frame / make_robot_action 都依赖这份 schema。
    # 根据 action/observation processor pipeline 构建一次 LeRobotDataset
    # schema，其中也包含图像/视频特征。
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc,
            initial_features=create_initial_features(observation=observation_features_hw),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    shutdown_event = Event()
    policy_active = Event()
    compile_warmup_done = Event()
    if not cfg.use_torch_compile:
        compile_warmup_done.set()
    rtc_thread = None

    try:
        # 1. 先准备 dataset 容器。此时还没连接硬件，失败成本最低。
        if cfg.resume:
            # LeRobotDataset.resume:
            # 输入: 已存在 dataset 的 repo_id/root，以及视频编码参数
            # 输出: 一个恢复到现有数据集上下文的 LeRobotDataset 对象
            # 作用: 继续往旧数据集追加 episode，而不是新建目录结构。
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot_raw.cameras if hasattr(robot_raw, "cameras") else []),
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
        else:
            # LeRobotDataset.create:
            # 输入: repo_id、fps、robot_type、dataset_features 以及编码参数
            # 输出: 一个全新的 LeRobotDataset 对象
            # 作用: 创建当前采集任务的数据集容器。后面 rollout 只会往它的
            # episode buffer 里 add_frame，真正保存由 save_episode() 完成。
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot_raw.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot_raw.cameras if hasattr(robot_raw, "cameras") else []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # 加载 policy：RTC 需要手动加载，以支持 predict_action_chunk。
        if use_rtc:
            # make_policy() 更偏向标准 select_action rollout。RTC 需要带额外
            # kwargs 调用 predict_action_chunk，所以这里手动组装 config。
            # get_policy_class:
            # 输入: cfg.policy.type，例如 "pi0" / "act" / "smolvla"
            # 输出: 对应策略类本身，而不是实例
            # 作用: 先根据字符串拿到具体 policy class，后面再 from_pretrained。
            policy_class = get_policy_class(cfg.policy.type)
            policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
            if hasattr(policy_config, "compile_model"):
                policy_config.compile_model = cfg.use_torch_compile
            # policy_class.from_pretrained:
            # 输入: 预训练模型路径 + policy config
            # 输出: 一个已加载权重但尚未开始 rollout 的 policy 实例
            # 作用: RTC 模式需要实例上暴露 predict_action_chunk，因此这里手动加载。
            policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=policy_config)
            policy.config.rtc_config = cfg.rtc
            if hasattr(policy, "init_rtc_processor"):
                policy.init_rtc_processor()
            policy = policy.to(cfg.device)
            policy.eval()
        else:
            # make_policy:
            # 输入: cfg.policy 和 dataset.meta
            # 输出: 一个标准的 PreTrainedPolicy 实例
            # 作用: 为同步 rollout 构造策略对象；内部会根据 policy 类型完成
            # from_pretrained、设备放置以及必要的包装。
            policy = make_policy(cfg.policy, ds_meta=dataset.meta)

        # make_pre_post_processors:
        # 输入: policy 配置、预训练路径、dataset 统计量，以及少量覆盖参数
        # 输出: (preprocessor, postprocessor)
        # 作用:
        # - preprocessor 把观测字典转成模型真正吃的张量格式
        # - postprocessor 把模型输出张量还原成机器人动作空间
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        # 2. 再连接硬件和 teleop。这样上面若 dataset/policy 初始化失败，
        # 就不会留下半连接状态的机器人设备。
        if use_rtc:
            logger.info("Connecting robot (calibrate=%s)", cfg.calibrate)
            # robot_raw.connect:
            # 输入: 是否需要在连接阶段触发校准
            # 输出: 无；副作用是打开总线/相机/硬件资源，使 robot 进入可读写状态
            # 作用: 从“配置对象”过渡到“真正在线的机械臂设备”。
            robot_raw.connect(calibrate=False)
            if cfg.calibrate and hasattr(robot_raw, "calibrate"):
                robot_raw.calibrate()
                robot_raw.disconnect()
                robot_raw.connect(calibrate=False)
        else:
            robot_raw.connect()

        # 这行是在统一“后续代码看到的 robot 接口”：
        # - 非 RTC: 直接使用原始 robot_raw，因为只有主线程访问它
        # - RTC: 用 ThreadSafeRobot 包一层，因为主 rollout 线程和后台线程会共享
        #   对机器人相关接口的访问，需要靠锁保证 get_observation/send_action 不并发冲突
        #
        # 这样后面的代码就不用分别写两套逻辑，而是统一调用：
        #   robot.get_observation()
        #   robot.send_action(action)
        robot = ThreadSafeRobot(robot_raw) if use_rtc else robot_raw
        # teleop.connect:
        # 输入: 无
        # 输出: 无；副作用是连接 leader/示教设备
        # 作用: 让后续 teleop.get_action() 能读到真实的人类输入。
        teleop.connect()
        # init_keyboard_listener:
        # 输入: 无
        # 输出: (listener, events)
        # 作用: 建立键盘到状态机事件的翻译层；rollout 主循环据此决定暂停、
        # 接管、恢复、保存或停止。
        listener, events = init_keyboard_listener()

        # 3. RTC 专用组件：动作队列、最新观测交接点、后台推理线程。
        queue_holder = None
        obs_holder = None
        obs_lock = Lock()
        hw_features = None
        if use_rtc:
            _start_pedal_listener(events)
            # ActionQueue:
            # 输入: cfg.rtc，尤其是 execution_horizon 等 RTC 参数
            # 输出: 一个动作队列对象
            # 作用: 后台推理线程把未来一段 action chunk 填进队列，
            # 主 rollout 循环再按控制节拍从队列里取动作执行。
            queue_holder = {"queue": ActionQueue(cfg.rtc)}
            # obs_holder 是主控制循环传递最新观测给 RTC 推理线程的交接点；
            # 访问时用 obs_lock 保护。
            obs_holder = {
                "obs": None,
                "robot_type": robot.robot_type,
                "action_feature_names": [key for key in robot.action_features if key.endswith(".pos")],
            }
            hw_features = hw_to_dataset_features(observation_features_hw, "observation")

            rtc_thread = Thread(
                target=_rtc_inference_thread,
                args=(
                    policy,
                    obs_holder,
                    obs_lock,
                    hw_features,
                    preprocessor,
                    postprocessor,
                    queue_holder,
                    shutdown_event,
                    policy_active,
                    compile_warmup_done,
                    cfg,
                ),
                daemon=True,
            )
            rtc_thread.start()

        print_controls(rtc=use_rtc)
        logger.info(f"  Policy: {cfg.policy.pretrained_path}")
        logger.info(f"  Task: {cfg.dataset.single_task}")
        logger.info(f"  Interpolation: {cfg.interpolation_multiplier}x")
        if use_rtc:
            logger.info(f"  RTC: enabled (execution_horizon={cfg.rtc.execution_horizon})")

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Episode {dataset.num_episodes}", cfg.play_sounds)

                # 4. 每次 rollout 都只往当前 episode buffer 里追加帧；
                # 是否 save/clear 由外层根据用户按键决定。
                if use_rtc:
                    queue_holder["queue"] = ActionQueue(cfg.rtc)
                    _rollout_rtc(
                        robot=robot,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=dataset,
                        events=events,
                        cfg=cfg,
                        queue_holder=queue_holder,
                        obs_holder=obs_holder,
                        obs_lock=obs_lock,
                        policy_active=policy_active,
                        compile_warmup_done=compile_warmup_done,
                        hw_features=hw_features,
                    )
                else:
                    _rollout_sync(
                        robot=robot,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=dataset,
                        events=events,
                        cfg=cfg,
                    )

                if events["rerecord_episode"]:
                    # 左箭头会丢弃当前 episode 的内存帧，并且不增加 recorded，
                    # 直接重新采这一条。
                    log_say("Re-recording", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # dataset.save_episode:
                # 输入: 这里不显式传 episode_data，表示直接把当前 episode buffer
                # 里已经累积好的帧落盘
                # 输出: 无；副作用是把这一条 episode 正式写入数据集，并清空当前
                # buffer 以便下一条 episode 继续记录。
                dataset.save_episode()
                recorded += 1

                if recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                    # save 之后才进入 reset，所以 reset 阶段的人工操作不会污染
                    # 刚保存完的 episode，也不会落入下一条 episode buffer。
                    reset_loop(robot, teleop, events, cfg.dataset.fps)

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        # 5. finally 是统一收尾阶段：即使用户提前退出或出现异常，也要清理
        # 硬件、视频编码器、监听线程，并按配置决定是否上传到 Hub。
        shutdown_event.set()
        policy_active.clear()

        if rtc_thread and rtc_thread.is_alive():
            rtc_thread.join(timeout=2.0)

        if dataset:
            dataset.finalize()

        if robot_raw.is_connected:
            robot_raw.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub and dataset is not None:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


def main():
    """注册第三方插件，并进入命令行入口。"""
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    hil_collect()


if __name__ == "__main__":
    main()
