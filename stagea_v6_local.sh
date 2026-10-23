#!/bin/bash
# Stage-A v6 on the local RTX 3060 (RESULTS.md section 7): v5 high-res branch with DINOv2 + 50/50 mixed sampling,
# all of OpenForensics (official Train / Val / Test-Dev / Test-Challenge), FakeClue FF++ frames and AI-generated /
# real people (more FF++ data for the type head). Batch 8 (12 GB card), resumable (last.pt every 1000 steps);
# 5 loader workers: 8 workers plus other jobs ran the 15 GB machine out of RAM at step 800.
source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=../data/index
CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python -W ignore -m forensics.train \
    --index $I/casia.jsonl $I/sd_inpaint.jsonl $I/ddvqa_c40.jsonl $I/openforensics_full.jsonl $I/diffusionface.jsonl \
            $I/fakeclue.jsonl \
    --roots casia=../data/CASIA2 sd_inpaint=../data/SD_inpaint ddvqa=./utils openforensics=../data/OpenForensics \
            diffusionface=../data/DiffusionFace/x fakeclue=../data/FakeClue \
    --m2f2-shard ./checkpoints/llava-v1.5-7b-M2F2-Det/model-00003-of-00003.safetensors \
    --out ./runs/stagea_v6 --steps 24000 --bs 8 --eval-every 4000 --save-every 1000 --eval-at-start --workers 5 \
    --distill 1.0 --distill-sources ddvqa fakeclue_ffpp --hr-size 672 --balance mix "$@"
