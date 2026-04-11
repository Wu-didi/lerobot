sudo chmod -R 777 /dev/ttyACM*
lerobot-record   \
 --robot.type=bi_koch_follower   \
 --robot.left_arm_port=/dev/ttyACM3   \
 --robot.right_arm_port=/dev/ttyACM4   \
 --robot.id=bimanual_follower   \
 --robot.cameras='{top: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.left: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}'   \
 --teleop.type=bi_koch_leader   \
 --teleop.left_arm_port=/dev/ttyACM1   \
 --teleop.right_arm_port=/dev/ttyACM2   \
 --teleop.id=bimanual_leader   \
 --display_data=false   \
 --dataset.repo_id=zsx/eval_fold_clothes \
 --dataset.num_episodes=500 \
 --dataset.single_task="Fold the T-shirt that's on the table." \
 --dataset.root=data/eval_test \
 --dataset.reset_time_s=1   \
 --dataset.episode_time_s=2000   \
 --policy.path=/media/wudi/f/wudi/lerobot/032000/pretrained_model \
 --policy.push_to_hub=false  \
 --dataset.push_to_hub=false 
#  --rtc.enabled=true \
#  --rtc.execution_horizon=20 \
#  --rtc.max_guidance_weight=5.0 \
#  --rtc.prefix_attention_schedule=LINEAR \
#  --interpolation_multiplier=3

