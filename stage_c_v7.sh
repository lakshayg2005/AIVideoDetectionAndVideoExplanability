#!/bin/bash
# Stage C v7 (RTX 8000, 48 GB): LoRA r=128 on the fp16 (not 4-bit) LLM, 3 epochs of the FF++-free text mix.
# Turing has no bf16, so fp16 AMP (trainable params fp32, see train_deepfake.py). Checkpoints every 200 steps, resumable.
source "${CONDA_BASE:-/data3/Aditya/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"
R=${R:-/data3/Aditya/bp_data}
CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python llava/train/train_deepfake.py \
    --bits 16 --attn_impl sdpa --optim adamw_torch \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path ${BASE:-$R/ckpt/forensic-base-v7} \
    --pretrain_deepfake_adapter $R/ckpt/stage-b-v7/mm_projector.bin \
    --version v1 \
    --data_path $R/data/stagec_v7/train_mix.json --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True --tune_deepfake_mlp_adapter True \
    --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --fp16 True --bf16 False --tf32 False \
    --output_dir $R/ckpt/stage-c-v7-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --evaluation_strategy no --save_strategy steps --save_steps 200 --save_total_limit 3 \
    --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 5 --model_max_length 2048 --gradient_checkpointing True \
    --dataloader_num_workers 8 --lazy_preprocess True --report_to none "$@"
