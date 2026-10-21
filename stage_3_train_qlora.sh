#!/bin/bash
# Stage 3 on one 12 GB GPU: QLoRA (4-bit NF4 base, LoRA r=128 like the paper) + paged 8-bit AdamW.
# Loads the stage-2 deepfake projector directly instead of merging a full fp16 model on CPU.
source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"

CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python llava/train/train_deepfake.py \
    --bits 4 --attn_impl sdpa --optim paged_adamw_8bit \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path ./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1 \
    --pretrain_deepfake_adapter ./checkpoints/llava-v1.5-7b-deepfake_stage-2-proj-qlora/mm_projector.bin \
    --version v1 \
    --data_path ./utils/DDVQA_split/c40/train_DDVQA_format.json \
    --image_folder ./utils/DDVQA_images/c40/train \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --deepfake_ckpt_path ./utils/weights/M2F2_Det_densenet121.pth \
    --tune_mm_mlp_adapter True --tune_deepfake_mlp_adapter True \
    --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --bf16 True --tf32 True \
    --output_dir ./checkpoints/llava-v1.5-7b-deepfake_stage-3-qlora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --evaluation_strategy no --save_strategy steps --save_steps 200 --save_total_limit 2 \
    --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 1 --model_max_length 2048 --gradient_checkpointing True \
    --dataloader_num_workers 4 --lazy_preprocess True --report_to tensorboard "$@"
