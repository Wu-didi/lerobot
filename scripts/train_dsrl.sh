#!/usr/bin/env bash
set -e

# 只基于已有 cache 训练离线 DSRL / latent IQL。
# 适用场景：cache 已经生成，后续只调整训练步数、batch size、学习率或 IQL/AWR 超参。
# 本脚本不会重新执行 action-to-noise 反演，因此不会重复消耗大量时间生成 cache。

# 训练会从 --cache_dir 读取 precompute_dsrl_cache.sh 生成的文件：
# - dsrl_offline_cache.safetensors：读取 noise_label、reward、done、return_to_go、dataset_index、next_dataset_index。
# - dsrl_offline_cache.json：读取 dataset_root、dataset_repo_id、return_to_go_mean/std 等 metadata。
# 训练目标：
# - critic 学习 Q(s, latent_noise)，目标是 sparse reward + discount * V(s')。
# - value 用 IQL expectile regression 拟合 demo latent action 的 conservative Q。
# - actor 按 advantage 权重拟合 noise_label，也就是学习更偏向高价值示范 latent noise。

# PYTHONPATH=src：继续使用当前仓库源码中的 DSRL 实现。
# /home/wudi/miniconda3/envs/lerobot/bin/python：继续固定使用同一个 conda 环境。
# -m lerobot.scripts.lerobot_train_offline_dsrl：运行离线 DSRL/IQL 训练入口。
# --policy.base_policy_path：同一个 base pi0.5 权重路径；训练时仍然冻结 base，只训练 actor/critic/value head。
# --policy.device：训练使用的设备；这里用 cuda 在 GPU 上编码观测并训练 DSRL head。
# --cache_dir：cache 目录；必须已经存在 dsrl_offline_cache.safetensors 和 dsrl_offline_cache.json。
# --output_dir：训练输出目录；会保存中间 checkpoint、final checkpoint、TensorBoard event 和训练状态 JSON。
# --batch_size：offline RL 训练 batch size；越大梯度越稳定但显存占用越高。
# --num_workers：训练 DataLoader worker 数；设为 0 便于排查数据读取和 processor 问题。
# --steps：总训练步数；这里训练 10000 个 optimizer step，不按 epoch 停止。
# --log_freq：每隔多少 step 刷新 tqdm/logging 指标；这里每 50 step 输出一次训练状态。
# --save_freq：每隔多少 step 保存一次中间 checkpoint；这里每 1000 step 保存一次。
# --seed：训练随机种子，控制 dataloader shuffle 和 DSRL head 初始化，方便复现实验。
PYTHONPATH=src /home/wudi/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_train_offline_dsrl \
    --policy.base_policy_path=/media/wudi/f/wudi/lerobot/pi05_training_dagger/012000/pretrained_model \
    --policy.device=cuda \
    --cache_dir=outputs/dsrl/fold_clothes_cache \
    --output_dir=outputs/dsrl/fold_clothes_iql \
    --batch_size=8 \
    --num_workers=0 \
    --steps=10000 \
    --log_freq=50 \
    --save_freq=1000 \
    --seed=0

# TensorBoard 可视化命令：
# /home/wudi/miniconda3/envs/lerobot/bin/tensorboard --logdir outputs/dsrl/fold_clothes_iql/runs
