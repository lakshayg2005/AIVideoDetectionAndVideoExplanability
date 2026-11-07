#!/bin/bash
# Stage B v7 (RTX 8000): frozen 16-bit (fp16) LLM instead of 4-bit, only the evidence projector trains. FF++-free mix.
source "${CONDA_BASE:-/data3/Aditya/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"
R=${R:-/data3/Aditya/bp_data}
CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python llava/train/train_deepfake.py \
    --bits 16 --attn_impl sdpa \
    --model_name_or_path ${BASE:-$R/ckpt/forensic-base-v7} \
    --version v1 \
    --data_path $R/data/stagec_v7/train_mix.json --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter False --freeze_mm_mlp_adapter True --freeze_backbone True \
    --tune_deepfake_mlp_adapter True \
    --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --fp16 True --bf16 False --tf32 False \
    --output_dir $R/ckpt/stage-b-v7 \
    --max_steps 300 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --evaluation_strategy no --save_strategy no \
    --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 1 --model_max_length 2048 --gradient_checkpointing True \
    --dataloader_num_workers 8 --lazy_preprocess True --report_to none "$@"
