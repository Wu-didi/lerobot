# 双臂 Koch 的 HIL / DAgger 数据采集脚本。
#
# 正常 rollout 流程：
#   1. 运行这个脚本，policy 自动控制 follower 机器人，并开始记录数据。
#   2. 当你觉得 policy 快要失败时，按 SPACE。
#      机器人会保持当前位置，leader 机械臂会自动移动到 follower 当前姿态。
#   3. 等 leader 对齐完成后，按 c 开始人工接管。
#   4. 用 leader 纠正 follower 的动作。这一段纠正数据会被记录下来。
#   5. 纠正完成后，按 p 把控制权交回 policy。
#      policy 会从纠正后的状态继续自动运行。
#
# episode 控制：
#   →    结束并保存当前 episode。
#   ←    放弃当前 episode，并重新录这一条。
#   ESC  停止整个采集流程。
#
# 按 → 之后怎么开始下一个 episode：
#   1. 当前 episode 会先被保存。
#   2. 程序进入 RESET 阶段，并把 leader 自动对齐到 follower 当前姿态。
#   3. 看到日志 "Press any key to enable teleoperation" 后，按 SPACE 或 →。
#   4. 这时可以用 leader 控制 follower，把机器人和场景摆到下一条的初始状态。
#      这个 reset 摆放过程不会被记录到数据集。
#   5. 看到日志 "Teleop enabled - press any key to start episode" 后，
#      再按一次 SPACE 或 →。
#      下一条 episode 正式开始，policy 重新接管机器人并开始记录数据。
#
# 如果在 RESET 阶段想停止整个采集，按 ESC。

sudo chmod -R 777 /dev/ttyACM*

python examples/hil/hil_data_collection.py \
 --rtc.enabled=true \
 --rtc.execution_horizon=20 \
 --rtc.max_guidance_weight=5.0 \
 --rtc.prefix_attention_schedule=LINEAR \
 --robot.type=bi_koch_follower \
 --robot.left_arm_port=/dev/ttyACM3 \
 --robot.right_arm_port=/dev/ttyACM4 \
 --robot.id=bimanual_follower \
 --robot.cameras='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.left: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.right: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
 --teleop.type=bi_koch_leader \
 --teleop.left_arm_port=/dev/ttyACM1 \
 --teleop.right_arm_port=/dev/ttyACM2 \
 --teleop.id=bimanual_leader \
 --policy.path=/media/wudi/f/wudi/lerobot/pi05_training_dagger/012000/pretrained_model  \
 --policy.push_to_hub=false \
 --dataset.repo_id=zsx/eval_fold_clothes \
 --dataset.root=data/eval_test_0504 \
 --dataset.single_task="Fold the T-shirt that's on the table." \
 --dataset.fps=30 \
 --dataset.episode_time_s=2000 \
 --dataset.num_episodes=500 \
 --dataset.push_to_hub=false \
 --display_data=false \
 --interpolation_multiplier=3 \
 --resume=true 
  #  --policy.path=/home/wudi/code/lerobot-0.4.2/data/pi05_training_dagger_0429/016000/pretrained_model \
  #  --policy.path=/media/wudi/f/wudi/lerobot/pi05_training_dagger/012000/pretrained_model \
  # /media/wudi/f/wudi/lerobot/pi05_training_v2/checkpoints/008000/pretrained_model