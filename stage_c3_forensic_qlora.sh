#!/bin/bash
# Stage C v3: same model and settings as stage_c_forensic_qlora.sh (v3 heads, Stage-B pilot projector), trained on
# the v3 text mix: as v2, plus hints naming the type head's verdict (forensics.texts.make_hint, 80% of fakes,
# also DDVQA) and every fake answer stating the forgery type (../data/stagec_v3/build.log). Checkpoints every 100 steps.
bash stage_c_forensic_qlora.sh \
    --pretrain_deepfake_adapter ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-b-v3pilot/mm_projector.bin \
    --data_path ../data/stagec_v3/train_mix.json \
    --output_dir ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-c3-lora \
    --save_steps 100 "$@"
