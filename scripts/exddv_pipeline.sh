#!/bin/bash
# ExDDV preprocessing, end to end: download videos -> extract fake frames + map text -> extract real frames -> validate.
# Safe to re-run: the download and finished videos are reused.
# Usage: bash scripts/exddv_pipeline.sh [extra exddv_preprocess.py args]
#   Watch progress from another terminal: tail -f data/ExDDV/pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/ExDDV
LOG=$OUT/pipeline.log
PY=${PY:-$HOME/miniforge3/envs/M2F2_det/bin/python}   # needs opencv 4.8+/5, pandas, numpy
DL_ENV=$OUT/.download_env                             # kagglehub stays out of M2F2_det's pinned env
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "PIPELINE_FAILED exit=$? at line $LINENO"' ERR
export OPENCV_LOG_LEVEL=ERROR

echo "=== $(date) download"
[ -x "$DL_ENV/bin/python" ] || { python3 -m venv "$DL_ENV"; "$DL_ENV/bin/pip" -q install kagglehub; }
DATASET=$("$DL_ENV/bin/python" scripts/exddv_download.py 2>"$OUT/download.log" | tail -1)
FFPP_ROOT=$DATASET/FaceForensics++_C23
echo "videos: $FFPP_ROOT ($(find "$FFPP_ROOT" -name '*.mp4' | wc -l) mp4 files)"

echo "=== $(date) preprocess"
"$PY" scripts/exddv_preprocess.py --ffpp-root "$FFPP_ROOT" --out "$OUT" "$@"

echo "=== $(date) real images"
"$PY" scripts/exddv_real.py --ffpp-root "$FFPP_ROOT" --out "$OUT"

echo "=== $(date) validate"
"$PY" scripts/exddv_validate.py --out "$OUT"
echo "PIPELINE_DONE $(date)"
