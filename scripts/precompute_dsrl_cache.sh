#!/usr/bin/env bash
set -e

# 只生成一次离线 DSRL latent cache。
# 适用场景：数据集、base pi0.5 权重、反演参数或 reward 设计发生变化时运行。
# 如果这些条件都没有变化，后续只需要运行 train_dsrl_from_cache.sh，不要重复运行本脚本。

# 本脚本会在 --output_dir 指定的目录下保存两类 cache 文件：
# 1. dsrl_offline_cache.safetensors：保存训练直接使用的 tensor 数据。
#    - noise_label：每个示范 action chunk 反演得到的 pi0.5 latent noise 标签，是 actor 学习的目标。
#    - recon_error：用 noise_label 解码回 action 后和原示范 action 的 MSE，用来判断反演质量。
#    - is_valid：全量反演结束后，根据 recon_error 分布求出的阈值统一判断；训练时只使用 True 的样本。
#    - dataset_index：当前 transition 对应的 LeRobotDataset 样本 index，也就是状态 s 的位置。
#    - next_dataset_index：下一个 latent decision 的样本 index，也就是状态 s' 的位置；terminal 时为 -1。
#    - episode_index：该 transition 属于哪一条 episode，方便后续排查 reward 或数据质量。
#    - reward：dsrl_pi0-style sparse reward；成功 episode 最后一步为 0，其余通常为 -1。
#    - done：当前 transition 是否为该 episode 的最后一个 query-step；用于 Bellman target 截断。
#    - return_to_go：从当前 query-step 到 episode 结束的折扣回报，AWR fallback 和统计会用到。
#    - absolute_index：原始数据表里的绝对 frame index，用于定位回原始数据。
#    - task_index：LeRobot 数据里的任务编号，多任务数据时用于区分任务。
# 2. dsrl_offline_cache.json：保存可读 metadata。
#    - dataset_repo_id / dataset_root / dataset_revision：生成 cache 时使用的数据集来源。
#    - query_stride：每隔多少 frame/chunk 采一个 latent decision transition。
#    - base_policy_path：反演 latent noise 时使用的 base pi0.5 权重路径。
#    - discount：生成 return_to_go 和训练 Bellman target 使用的折扣因子。
#    - latent_recon_threshold：最终实际采用的重建误差阈值；默认由全量 recon_error 分位数自动求出。
#    - latent_recon_quantile / recon_error_p50/p90/p95/p99：用于复查阈值选择是否合理的统计量。
#    - num_total_samples / num_valid_samples：cache 总样本数和可训练样本数。
#    - return_to_go_mean / return_to_go_std：return 归一化统计，AWR fallback 使用。

# PYTHONPATH=src：让 Python 优先使用当前仓库里的 lerobot 源码，而不是环境里可能已安装的旧版本。
# /home/wudi/miniconda3/envs/lerobot/bin/python：固定使用 lerobot 这个 conda 环境的 Python，避免依赖版本不一致。
# -m lerobot.scripts.lerobot_precompute_dsrl_offline_cache：运行离线 DSRL cache 生成入口。
# --dataset.repo_id：LeRobot 数据集的 repo id，会用于读取数据集 metadata、feature 定义和任务信息。
# --dataset.root：本地数据集目录，指向你已经合并好的折衣服离线示范数据。
# --dataset.batch_size：反演 latent noise 时的 batch size；越大越快但显存占用越高。
# --dataset.num_workers：DataLoader worker 数；设为 0 方便调试真实数据读取问题，也避免多进程额外占显存。
# --policy.base_policy_path：已经训练好的 base pi0.5 policy 权重路径；DSRL 会冻结它，只把它当 latent-noise 解码器。
# --policy.device：模型和反演计算使用的设备；这里用 cuda 在 GPU 上执行 pi0.5 编码、解码和 latent 优化。
# --policy.latent_inversion_steps：每个样本 action-to-noise 反演的最大 Adam 优化步数，也就是 max_steps；全量首版先用较小值控制时间。
# --policy.latent_inversion_patience：早停 patience；连续若干步 improvement 小于 min_delta 就停止当前 restart。
# --policy.latent_inversion_min_delta：早停的最小有效改善量；小于这个值认为反演已经基本没有进步。
# --policy.latent_inversion_decode_steps：反演专用 pi0.5 denoise 步数；每个 Adam step 都要解码一次，所以这里要明显小于正式推理步数。
# --policy.latent_inversion_log_freq：反演内层日志频率；每隔若干 Adam step 打印 loss/recon_error，用来看是否下降。
# --policy.latent_restarts：每个样本从多少个随机 latent 初值重复反演；取误差最小的结果，降低局部最优风险。
# --policy.latent_recon_quantile：全量反演完成后，用该分位数自动求 recon_error 阈值；0.95 表示约保留误差最低的 0.95 比例样本。
# --policy.latent_recon_max_threshold：自动阈值的可选绝对上限；需要防止阈值过松时再打开。
# --output_dir：cache 输出目录，会保存 dsrl_offline_cache.safetensors 和 dsrl_offline_cache.json。
# --seed：随机种子，控制 latent 初始化、数据加载等随机过程，方便复现实验。
PYTHONPATH=src /home/wudi/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_precompute_dsrl_offline_cache \
    --dataset.repo_id=zsx/fold_clothes0402 \
    --dataset.root=/home/wudi/code/lerobot-0.4.2/data/fold_clothes_merged_all_0423_wo_online_dagger \
    --dataset.batch_size=4 \
    --dataset.num_workers=0 \
    --policy.base_policy_path=/media/wudi/f/wudi/lerobot/pi05_training_dagger/012000/pretrained_model \
    --policy.device=cuda \
    --policy.latent_inversion_steps=60 \
    --policy.latent_inversion_patience=6 \
    --policy.latent_inversion_min_delta=1e-5 \
    --policy.latent_inversion_decode_steps=10 \
    --policy.latent_inversion_log_freq=1 \
    --policy.latent_restarts=1 \
    --policy.latent_recon_quantile=0.95 \
    --policy.latent_recon_max_threshold=0.05 \
    --output_dir=outputs/dsrl/fold_clothes_cache \
    --seed=0
