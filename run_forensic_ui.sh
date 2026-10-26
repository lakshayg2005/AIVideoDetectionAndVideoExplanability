#!/bin/bash
# Explainable deepfake UI: forensic heatmaps + verdict + type + filters, and LLM explanations of the marked regions.
# Usage: bash run_forensic_ui.sh              # full (GPU), http://127.0.0.1:7861
#        bash run_forensic_ui.sh --no-llm     # heatmaps only on the CPU (e.g. while the GPU trains)
#        bash run_forensic_ui.sh --share      # also a public gradio.live link
#        bash run_forensic_ui.sh --stage-c2   # explanations by the Stage C v2 explainer (RESULTS.md 8)
#        bash run_forensic_ui.sh --stage-c3   # Stage C v3: hints and answers name the forgery type
source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"
ARGS=()
for a in "$@"; do
    if [ "$a" = --stage-c3 ]; then
        ARGS+=(--self-explain --hint-style v3 --model-path ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-c3-lora
               --model-base ./checkpoints/llava-v1.5-7b-M2F2-forensic-base)
    elif [ "$a" = --stage-c2 ]; then
        ARGS+=(--self-explain --model-path ./checkpoints/llava-v1.5-7b-M2F2-forensic_stage-c2-lora
               --model-base ./checkpoints/llava-v1.5-7b-M2F2-forensic-base)
    else
        ARGS+=("$a")
    fi
done
CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python -W ignore -m app.forensic_webui "${ARGS[@]}"
