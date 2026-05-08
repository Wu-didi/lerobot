sudo chmod -R 777 /dev/ttyACM*

# 数据采集
record_args=(
  # 机器人本体类型：双臂 Koch 从臂/执行端
  --robot.type=bi_koch_follower
  # 左侧从臂串口设备路径
  --robot.left_arm_port=/dev/ttyACM3
  # 右侧从臂串口设备路径
  --robot.right_arm_port=/dev/ttyACM4
  # 机器人实例 ID，用于区分不同硬件配置或日志记录
  --robot.id=bimanual_follower
  # 相机配置：top、wrist.left、wrist.right 三路 OpenCV 相机，指定设备索引、分辨率、帧率和编码格式
  '--robot.cameras={top: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.left: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, wrist.right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}'
  # 遥操作设备类型：双臂 Koch 主臂/示教端
  --teleop.type=bi_koch_leader
  # 左侧主臂串口设备路径
  --teleop.left_arm_port=/dev/ttyACM1
  # 右侧主臂串口设备路径
  --teleop.right_arm_port=/dev/ttyACM2
  # 遥操作设备实例 ID
  --teleop.id=bimanual_leader
  # 是否实时显示采集到的图像和状态数据
  --display_data=false
  # 数据集仓库 ID，通常用于 Hugging Face Hub 命名
  --dataset.repo_id=wudi/fold_clothes_0415_offline
  # 计划采集的 episode 数量
  --dataset.num_episodes=2500
  # 当前数据集对应的自然语言任务描述
  "--dataset.single_task=Fold the T-shirt that's on the table."
  # 数据集本地保存目录
  --dataset.root=data/fold_clothes_0415_offline_dagger
  # 每个 episode 结束后的复位/整理等待时间，单位秒
  --dataset.reset_time_s=20
  # 单个 episode 的最长录制时间，单位秒
  --dataset.episode_time_s=2000
  # 是否采集完成后上传到 Hugging Face Hub
  --dataset.push_to_hub=false
  # 是否使用流式编码保存数据；关闭时通常在本地按普通方式编码/写入
  --dataset.streaming_encoding=false
  # 是否从已有数据集进度继续采集，避免从头开始
  --resume=true
)

lerobot-record "${record_args[@]}"
# 是否启用 DAgger 数据采集模式；当前被注释掉，表示不启用
#  --dagger=true

