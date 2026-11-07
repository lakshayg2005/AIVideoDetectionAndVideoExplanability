#!/bin/bash
# Stage-A v7 on the Quadro RTX 8000 (48 GB). v6 recipe (DINOv2 hi-res branch, 50/50 mixed sampling, all OpenForensics
# splits) WITHOUT any FaceForensics++-derived data (DDVQA, FakeClue ff++): FF++ is kept for testing. Hence no
# distillation (it only applied to DDVQA). Resumable: last.pt every 1000 steps. BS default 24 (use BS=12 if the GPU is shared).
source "${CONDA_BASE:-/data3/Aditya/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${D:-/data3/Aditya/bp_data/data}
W=${W:-/data3/Aditya/bp_data/weights/llava-v1.5-7b-M2F2-Det}
I=$D/index
CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python -W ignore -m forensics.train \
    --index $I/casia.jsonl $I/sd_inpaint.jsonl $I/openforensics_full.jsonl $I/diffusionface.jsonl $I/fakeclue_noffpp.jsonl \
    --roots casia=$D/CASIA2 sd_inpaint=$D/SD_inpaint openforensics=$D/OpenForensics \
            diffusionface=$D/DiffusionFace/x fakeclue=$D/FakeClue \
    --m2f2-shard $W/model-00003-of-00003.safetensors \
    --out ${OUT:-/data3/Aditya/bp_data/runs/stagea_v7} --steps ${STEPS:-40000} --bs ${BS:-24} --lr 4e-4 \
    --eval-every 4000 --save-every 250 --eval-at-start --workers ${WORKERS:-16} --val-workers 8 \
    --hr-size 672 --balance mix "$@"
