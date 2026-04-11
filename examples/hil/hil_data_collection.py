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


# RTC 辅助函数


class ThreadSafeRobot:
    """Thread-safe wrapper for robot operations (used with RTC background thread)."""

    def __init__(self, robot: Robot):
        self._robot = robot
        self._lock = Lock()

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self._robot.get_observation()

    def send_action(self, action: dict) -> None:
        with self._lock:
            self._robot.send_action(action)

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})


def _set_openarm_max_relative_target_if_missing(
    robot_cfg: RobotConfig, max_relative_target: float = 8.0
) -> None:
    """Set a safe default max_relative_target for OpenArm followers when not provided."""
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
    """Convert absolute leftovers into model space for relative-action RTC policies."""
    if relative_step is None:
        return prev_actions_absolute.to(policy_device)

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
    """Pad/truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected prev_actions to be 2D [T, A], got shape={tuple(prev_actions.shape)}")

    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]

    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


def _resolve_action_key_order(cfg, dataset_action_names: list[str]) -> list[str]:
    """Choose action name ordering used to map policy tensor outputs to robot action dict."""
    policy_action_names = getattr(cfg.policy, "action_feature_names", None)
    if not policy_action_names:
        return dataset_action_names

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
    """Resolve joint-state ordering used to build observation.state."""
    if not policy_action_names:
        return available_joint_names

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
    """Start foot pedal listener thread if evdev is available.

    Pedal input is restricted to HIL control handoff only:
    policy -> pause -> takeover -> resume policy.
    Episode save/advance remains keyboard-only (right arrow).
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
                        events["resume_policy"] = True
                    elif events["policy_paused"]:
                        events["start_next_episode"] = True
                    else:
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
    """Background thread for RTC action chunk generation."""
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
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: HILDatasetConfig
    policy: PreTrainedConfig | None = None
    rtc: RTCConfig = field(default_factory=RTCConfig)
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
    """Rollout loop with standard synchronous inference."""
    fps = cfg.dataset.fps
    device = get_safe_torch_device(cfg.device)
    stream_online = bool(cfg.dataset.streaming_encoding)
    record_stride = 1 if cfg.record_interpolated_actions else max(1, cfg.interpolation_multiplier)

    # episode 边界处重置 policy 和 processor，避免循环状态或动作队列
    # 泄漏到下一条 rollout。
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    frame_buffer: list[dict] = []
    teleop_disable_torque(teleop)

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

    timestamp = 0.0
    record_tick = 0
    start_t = time.perf_counter()
    stats_window_start = start_t
    policy_inference_count = 0
    robot_command_count = 0

    while timestamp < cfg.dataset.episode_time_s:
        loop_start = time.perf_counter()

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
            # 把控制权交还给 policy 时，重新开始一段干净的动作上下文。
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
            # 每次按暂停只执行一次：先把 teleop 对齐到当前机器人姿态，
            # 再等待人工接管或恢复 policy。
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
            # 这里 start_next_episode 表示"开始接管"，因为暂停状态下
            # 键盘监听器会把 c 键映射到这个标志。
            teleop_disable_torque(teleop)
            events["start_next_episode"] = False
            events["correction_active"] = True
            waiting_for_takeover = False

        obs = robot.get_observation()
        obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
        obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
        obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)

        if events["correction_active"]:
            # 人工纠正动作会作为这一帧的训练 action 被记录下来。
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
            # 暂停时重复发送上一条 policy action，用于保持当前位置。
            if last_action:
                robot.send_action(last_action)
                robot_command_count += 1

        else:
            if interpolator.needs_new_action():
                # 同步模式会在这里阻塞等待 policy 返回下一条动作，
                # 因此更适合推理较快的策略。
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
                robot_action = make_robot_action(action_values, dataset.features)
                action_tensor = torch.tensor([robot_action[k] for k in action_keys])
                interpolator.add(action_tensor)

            interp_action = interpolator.get()
            if interp_action is not None:
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

        if cfg.display_data and robot_action:
            log_rerun_data(observation=obs_filtered, action=robot_action)

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

    teleop_disable_torque(teleop)

    if not stream_online:
        for frame in frame_buffer:
            dataset.add_frame(frame)


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
    """Rollout loop with RTC for asynchronous inference."""
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

    robot_action: dict[str, Any] = {}
    timestamp = 0.0
    start_t = time.perf_counter()
    stats_window_start = start_t
    robot_command_count = 0
    record_tick = 0
    obs_poll_interval = 1.0 / fps
    last_obs_poll_t = 0.0
    obs_filtered: dict[str, Any] = {}
    obs_frame: dict[str, Any] = {}
    warmup_wait_logged = False
    warmup_queue_flushed = False

    while timestamp < cfg.dataset.episode_time_s:
        loop_start = time.perf_counter()

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
            # 先停掉后台推理，再把 teleop 移动到机器人姿态，
            # 然后等待人工纠正或恢复 policy。
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
            obs = robot.get_observation()
            obs_filtered = {k: obs[k] for k in obs_state_names if k in obs}
            obs_filtered.update({k: obs[k] for k in obs_image_names if k in obs})
            obs_frame = build_dataset_frame(dataset.features, obs_filtered, prefix=OBS_STR)
            with obs_lock:
                obs_holder["obs"] = obs_filtered
            last_obs_poll_t = now_for_obs

        if events["correction_active"]:
            # 纠正模式下，teleop action 会被执行，同时也会写入数据集。
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
            # 没有控制器产生下一条动作时，继续发送上一条动作以保持机器人稳定。
            if last_action:
                robot.send_action(last_action)
                robot_command_count += 1

        else:
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
                    new_action = queue.get() if queue else None
                    if new_action is not None:
                        interpolator.add(new_action.cpu())

                action_tensor = interpolator.get()
                if action_tensor is not None:
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

    policy_active.clear()
    teleop_disable_torque(teleop)

    if not stream_online:
        for frame in frame_buffer:
            dataset.add_frame(frame)


# 主采集函数


@parser.wrap()
def hil_collect(cfg: HILConfig) -> LeRobotDataset:
    """Main HIL data collection function (supports both sync and RTC modes)."""
    init_logging()
    logger.info(pformat(cfg.__dict__))

    use_rtc = cfg.rtc.enabled

    if use_rtc:
        _set_openarm_max_relative_target_if_missing(cfg.robot, max_relative_target=8.0)

    if cfg.display_data:
        init_rerun(session_name="hil_collection")

    robot_raw = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

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
        if cfg.resume:
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
            policy_class = get_policy_class(cfg.policy.type)
            policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
            if hasattr(policy_config, "compile_model"):
                policy_config.compile_model = cfg.use_torch_compile
            policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=policy_config)
            policy.config.rtc_config = cfg.rtc
            if hasattr(policy, "init_rtc_processor"):
                policy.init_rtc_processor()
            policy = policy.to(cfg.device)
            policy.eval()
        else:
            policy = make_policy(cfg.policy, ds_meta=dataset.meta)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        # 连接硬件
        if use_rtc:
            logger.info("Connecting robot (calibrate=%s)", cfg.calibrate)
            robot_raw.connect(calibrate=False)
            if cfg.calibrate and hasattr(robot_raw, "calibrate"):
                robot_raw.calibrate()
                robot_raw.disconnect()
                robot_raw.connect(calibrate=False)
        else:
            robot_raw.connect()

        robot = ThreadSafeRobot(robot_raw) if use_rtc else robot_raw
        teleop.connect()
        listener, events = init_keyboard_listener()

        # RTC 专用初始化
        queue_holder = None
        obs_holder = None
        obs_lock = Lock()
        hw_features = None
        if use_rtc:
            _start_pedal_listener(events)
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

                # 每次 rollout 都会填充 dataset 的 episode buffer；
                # 返回后由外层决定保存还是清空。
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

                dataset.save_episode()
                recorded += 1

                if recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                    reset_loop(robot, teleop, events, cfg.dataset.fps)

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        # 即使用户提前退出或出现异常，也要清理硬件、视频编码器、
        # 监听线程，并按配置决定是否上传到 Hub。
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
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    hil_collect()


if __name__ == "__main__":
    main()
