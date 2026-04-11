### 注意事项

1. 夹取物体重量尽量不要超过100g
2. 机械臂闲置的时候拔掉电源
3. 机械臂因为过载或者代码出bug导致爆红灯、不工作，要拔插机械臂的电源

### 代码适配

#### 1. 支持koch双臂

解压code.zip，

1. bi_koch_follower文件夹放入src/lerobot/robots

2. bi_koch_leader文件夹放入lerobot/teleoperators

3. 作以下代码改动

   ```python
   # 1. src/lerobot/robots/utils.py 中，make_robot_from_config 函数中增加
   elif config.type == "bi_koch_follower":
       from .bi_koch_follower import BiKochFollower
   
       return BiKochFollower(config)
   
   # 2. src/lerobot/teleoperators/utils.py 中，def make_teleoperator_from_config 函数中增加
   elif config.type == "bi_koch_leader":
       from .bi_koch_leader import BiKochLeader
   
       return BiKochLeader(config)
       
   # 3. src/lerobot/scripts/lerobot_record.py 和 src/lerobot/scripts/lerobot_teleoperate.py中增加
   from lerobot.robots import bi_koch_follower
   from lerobot.teleoperators import bi_koch_leader
   ```

#### 2. lerobot框架对koch机械臂的适配BUG解决

**BUG 1**：有时候会报类似错误：Failed to sync read 'Present_Position' on ids=[2,3,4,6]after 1 tries. [TxRxResult] There is no status packet

这个是read的错误，有时候write的时和也会有类似错误，这不是koch的问题，是lerobot的bug，不知道现在有没有解决，解决方案就是在对应的read或者write部分增加num_retry参数，要是30还不够就继续增加，这个代码要自己根据报错往上找，一般就是self.bus.sync_read，self.sync_read，self.write等，例如

```python
# lerobot/robots/koch_follower/koch_follower.py中，obs_dict = self.bus.sync_read("Present_Position")改为
obs_dict = self.bus.sync_read("Present_Position", num_retry=30)

# 其它地方写
self.write("Homing_Offset", motor, 0, normalize=False, num_retry=30)
```

**BUG2**：进行遥操或者采集数据或者推理的时候，从臂的夹爪会自动转圈，这也是lerobot的bug，解决方式是断电后，把主、从臂的夹爪都往左转一点，尽量角度一样，有时候得多试几次才行。

### 相关命令举例

直接运行遥远命令，没标定的话会自动出发标定流程的，不用单独运行lerobot_calibrate；

参考命令修改port和camera的数量和index，camera的"fourcc": "MJPG"这部分不要动

```bash
# 遥操
lerobot-teleoperate \
  --robot.type=bi_koch_follower \
  --robot.left_arm_port=/dev/ttyACM2 \
  --robot.right_arm_port=/dev/ttyACM4 \
  --robot.id=bimanual_follower \
  --robot.cameras='{top: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}'   \
  --teleop.type=bi_koch_leader \
  --teleop.left_arm_port=/dev/ttyACM3 \
  --teleop.right_arm_port=/dev/ttyACM0 \
  --teleop.id=bimanual_leader \
  --display_data=true

# 数据采集
lerobot-record   \
 --robot.type=bi_koch_follower   \
 --robot.left_arm_port=/dev/ttyACM4   \
 --robot.right_arm_port=/dev/ttyACM2   \
 --robot.id=bimanual_follower   \
 --robot.cameras='{top: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.left: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.right: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}'   \
 --teleop.type=bi_koch_leader   \
 --teleop.left_arm_port=/dev/ttyACM3   \
 --teleop.right_arm_port=/dev/ttyACM1   \
 --teleop.id=bimanual_leader   \
 --display_data=false   \
 --dataset.repo_id=zsx/fold_clothes   \
 --dataset.num_episodes=2500   \
 --dataset.single_task="Fold the T-shirt that's on the table."   \
 --dataset.root=data/fold_clothes   \
 --dataset.reset_time_s=1   \
 --dataset.episode_time_s=2000   \
 --resume=true \
 --dagger=true

# 训练
lerobot-train \
    --dataset.repo_id=zsx/fold_clothes \
    --dataset.root=data/fold_clothes \
    --policy.type=pi05 \
    --output_dir=outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=zsx/pi05 \
    --policy.push_to_hub=false \
    --policy.pretrained_path=./pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.training_rtc=true  \
    --policy.dtype=bfloat16 \
    --wandb.enable=false \
    --steps=100000 \
    --policy.device=cuda \
    --batch_size=128 \
    --log_freq=100 \
    --save_freq=4000


# 测试
lerobot-record   \
 --robot.type=bi_koch_follower   \
 --robot.left_arm_port=/dev/ttyACM4   \
 --robot.right_arm_port=/dev/ttyACM2   \
 --robot.id=bimanual_follower   \
 --robot.cameras='{top: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.left: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.right: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}'   \
 --teleop.type=bi_koch_leader   \
 --teleop.left_arm_port=/dev/ttyACM3   \
 --teleop.right_arm_port=/dev/ttyACM1   \
 --teleop.id=bimanual_leader   \
 --display_data=false   \
 --dataset.repo_id=zsx/eval_fold_clothes \
 --dataset.num_episodes=500 \
 --dataset.single_task="Fold the T-shirt that's on the table." \
 --dataset.root=data/eval_test \
 --dataset.reset_time_s=1   \
 --dataset.episode_time_s=2000   \
 --policy.path=outputs/pretrained_model
```

