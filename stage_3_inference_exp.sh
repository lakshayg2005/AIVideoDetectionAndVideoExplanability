#!/bin/bash
source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=0
CUDA_VISIBLE_DEVICES=$CUDA_NUM python -m llava.serve.cli_DDVQA_exp \
    --model-path ./checkpoints/llava-v1.5-7b-M2F2-Det \
    --load-4bit