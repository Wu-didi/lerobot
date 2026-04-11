# lerobot-train \
#     --dataset.repo_id=zsx/fold_clothes0402 \
#     --dataset.root=./data/fold_clothes0402 \
#     --policy.type=pi05 \
#     --output_dir=outputs/pi05_training \
#     --job_name=pi05_training \
#     --policy.repo_id=zsx/pi05 \
#     --policy.push_to_hub=false \
#     --policy.pretrained_path=./pi05_base \
#     --policy.compile_model=false \
#     --policy.gradient_checkpointing=true \
#     --policy.dtype=bfloat16 \
#     --wandb.enable=false \
#     --steps=100000 \
#     --policy.device=cuda \
#     --batch_size=32 \
#     --log_freq=100 \
#     --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --save_freq=4000



  lerobot-train \
      --dataset.repo_id=zsx/fold_clothes0402 \
      --dataset.root=/home/wudi/code/lerobot_0.5.1/lerobot/data/eval_test \
      --policy.type=pi05 \
      --output_dir=outputs/pi05_training_lora \
      --job_name=pi05_training \
      --policy.repo_id=zsx/pi05 \
      --policy.push_to_hub=false \
      --policy.pretrained_path=/media/wudi/f/wudi/lerobot/032000/pretrained_model \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=true \
      --policy.dtype=bfloat16 \
      --wandb.enable=false \
      --steps=100000 \
      --policy.device=cuda \
      --batch_size=16 \
      --log_freq=100 \
      --save_freq=1000 \
      --policy.optimizer_lr=2.5e-4 \
      --policy.scheduler_decay_lr=2.5e-5 \
      --peft.method_type=LORA \
      --peft.r=64
