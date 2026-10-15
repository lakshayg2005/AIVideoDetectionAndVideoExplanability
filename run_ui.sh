#!/bin/bash
# Launch the M2F2-Det web UI (image/video upload -> verdict + explanation).
# Usage: bash run_ui.sh            # real model, http://127.0.0.1:7860
#        bash run_ui.sh --share    # also prints a public gradio.live link
#        bash run_ui.sh --mock     # UI only, no GPU/weights
source /home/user2/miniforge3/bin/activate M2F2_det
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=0
CUDA_VISIBLE_DEVICES=$CUDA_NUM python -m app.webui \
    --model-path ./checkpoints/llava-v1.5-7b-M2F2-Det "$@"
