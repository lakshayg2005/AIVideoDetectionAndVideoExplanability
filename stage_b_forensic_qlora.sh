#!/bin/bash
# Stage B with the Stage-A forensic heads: 4-bit frozen M2F2 LLM, train only the new evidence projector
# (16-dim evidence token; region tokens reuse the trained mm_projector). PLAN.md.
# Needs: scripts/prepare_forensic_base.py, scripts/merge_stagec.py
source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"

CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python llava/train/train_deepfake.py \
    --bits 4 --attn_impl sdpa \
    --model_name_or_path ./checkpoints/llava-v1.5-7b-M2F2-forensic-base \
    --version v1 \
    --data_path ../data/stagec/train_mix.json --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter False --freeze_mm_mlp_adapter True --freeze_backbone True \
    --tune_deepfake_mlp_adapter True \
    --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --bf16 True --tf32 True \
    --output_dir ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-b \
    --max_steps 150 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
    --evaluation_strategy no --save_strategy no \
    --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 1 --model_max_length 2048 --gradient_checkpointing True \
    --dataloader_num_workers 4 --lazy_preprocess True --report_to tensorboard "$@"
