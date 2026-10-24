#!/bin/bash
# Stage C v2: same model and settings as stage_c_forensic_qlora.sh (v3 heads, Stage-B pilot projector), trained on
# the v2 text mix (forensics/texts.py: teacher region names, heatmap hints in half the questions, lighting clauses
# thinned, FakeClue; ../data/stagec_v2/build.log). Checkpoints every 100 steps (~1.5 h); resumes from the newest.
bash stage_c_forensic_qlora.sh \
    --pretrain_deepfake_adapter ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-b-v3pilot/mm_projector.bin \
    --data_path ../data/stagec_v2/train_mix.json \
    --output_dir ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-c2-lora \
    --save_steps 100 "$@"
