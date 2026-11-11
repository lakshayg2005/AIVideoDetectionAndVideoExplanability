#!/bin/bash
# Explainable deepfake UI: forensic heatmaps + verdict + type + filters, and LLM explanations of the marked regions.
# Usage: bash run_forensic_ui.sh              # full (GPU), http://127.0.0.1:7861
#        bash run_forensic_ui.sh --no-llm     # heatmaps only on the CPU (e.g. while the GPU trains)
#        bash run_forensic_ui.sh --share      # also a public gradio.live link
#        bash run_forensic_ui.sh --stage-c2   # explanations by the Stage C v2 explainer (RESULTS.md 8)
#        bash run_forensic_ui.sh --stage-c3   # Stage C v3: hints and answers name the forgery type
#        bash run_forensic_ui.sh --stage-c7   # v7: high-res heads + Stage C explainer in 16-bit (~17 GB GPU), no FF++ in training
#        add --host 0.0.0.0 to listen on the network (no password!); the default 127.0.0.1 is reached through an SSH tunnel:
#        ssh -L 7861:127.0.0.1:7861 <user>@<server>   then open http://127.0.0.1:7861
CB="${CONDA_BASE:-$HOME/miniconda3}"; [ -f "$CB/bin/activate" ] || CB=/data3/Aditya/miniconda3
source "$CB/bin/activate" M2F2_det
V7="${BP_DATA:-/data3/Aditya/bp_data}"
export PYTHONPATH="$(pwd):$PYTHONPATH"
ARGS=()
for a in "$@"; do
    if [ "$a" = --stage-c3 ]; then
        ARGS+=(--self-explain --hint-style v3 --model-path ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-c3-lora
               --model-base ./checkpoints/llava-v1.5-7b-M2F2-forensic-base)
    elif [ "$a" = --stage-c7 ]; then
        ARGS+=(--self-explain --hint-style v3 --fp16 --stagea "$V7/runs/stagea_v7/last.pt"
               --m2f2-shard "$V7/weights/llava-v1.5-7b-M2F2-Det/model-00003-of-00003.safetensors"
               --model-path "$V7/ckpt/llava-v1.5-7b-M2F2-forensic_stage-c7-lora" --model-base "$V7/ckpt/forensic-base-v7")
    elif [ "$a" = --stage-c2 ]; then
        ARGS+=(--self-explain --model-path ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-c2-lora
               --model-base ./checkpoints/llava-v1.5-7b-M2F2-forensic-base)
    else
        ARGS+=("$a")
    fi
done
CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python -W ignore -m app.forensic_webui "${ARGS[@]}"
