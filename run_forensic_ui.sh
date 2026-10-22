#!/bin/bash
# Explainable deepfake UI: forensic heatmaps + verdict + type + filters, and LLM explanations of the marked regions.
# Usage: bash run_forensic_ui.sh              # full (GPU), http://127.0.0.1:7861
#        bash run_forensic_ui.sh --no-llm     # heatmaps only on the CPU (e.g. while the GPU trains)
#        bash run_forensic_ui.sh --share      # also a public gradio.live link
source "${CONDA_BASE:-$HOME/miniconda3}/bin/activate" M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=${CUDA_NUM:-0} python -W ignore -m app.forensic_webui "$@"
