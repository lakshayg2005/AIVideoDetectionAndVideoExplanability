#!/bin/bash
# Whole pipeline on the RTX 8000, unattended and resumable (re-run it after any interruption; finished steps are skipped):
#   wait for data extraction + a free GPU -> Stage A (v7 heads) -> forensic base -> Stage B -> Stage C -> evaluation.
# No FaceForensics++-derived training data (DDVQA is only used as a held-out TEST set in the final evaluation).
R=/data3/Aditya/bp_data; L=$R/logs; cd "$(dirname "$0")"
source /data3/Aditya/miniconda3/bin/activate M2F2_det
export PYTHONPATH="$(pwd):$PYTHONPATH"
NEED_MB=${NEED_MB:-30000}
run_stage() {   # run_stage <name> <log> <cmd...>: wait for the GPU, run, retry (up to 30x) if killed; stages resume from checkpoints
  name=$1; log=$2; shift 2
  for try in $(seq 1 30); do
    wait_gpu; say "$name start (try $try)"
    "$@" >> $log 2>&1 && return 0
    say "$name failed or was killed (try $try); re-queueing in 30 s"; sleep 30
  done
  say "$name gave up after 30 tries"; return 1
}
say() { echo "[$(date '+%F %T')] $*" | tee -a $L/pipeline.log; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge $NEED_MB ]; do sleep 5; done; }

say "waiting for data extraction"
until grep -q OF_EXTRACT_DONE $L/ex_of.log && grep -q SD_EXTRACT_DONE $L/ex_sd.log && grep -q FC_EXTRACT_DONE $L/ex_fc.log; do sleep 30; done

if [ ! -f $R/runs/stagea_v7/DONE ]; then
  run_stage "Stage A" $L/stagea_v7.log bash stagea_v7_rtx8000.sh || exit 1
  touch $R/runs/stagea_v7/DONE
fi
if [ ! -f $R/ckpt/forensic-base-v7/config.json ]; then
  say "prepare forensic base"
  python scripts/prepare_forensic_base.py --m2f2-path $R/weights/llava-v1.5-7b-M2F2-Det \
      --stagea $R/runs/stagea_v7/last.pt --out $R/ckpt/forensic-base-v7 >> $L/prepare_base.log 2>&1 || { say "prepare failed"; exit 1; }
fi
if [ ! -f $R/ckpt/stage-b-v7/mm_projector.bin ]; then
  run_stage "Stage B" $L/stageb_v7.log bash stage_b_v7.sh || exit 1
fi
if [ ! -f $R/ckpt/stage-c-v7-lora/adapter_model.safetensors ]; then
  run_stage "Stage C" $L/stagec_v7.log bash stage_c_v7.sh || exit 1
fi
if [ ! -f $R/eval_v7/summary.json ]; then
  wait_gpu; say "evaluation start"
  I=$R/data/index; D=$R/data
  python -W ignore -m explain.eval_selfexplain --model-path $R/ckpt/stage-c-v7-lora --model-base $R/ckpt/forensic-base-v7 \
      --index $I/ddvqa_c40.jsonl $I/diffusionface.jsonl $I/casia.jsonl $I/sd_inpaint.jsonl $I/openforensics_full.jsonl \
      --roots ddvqa=./utils diffusionface=$D/DiffusionFace/x casia=$D/CASIA2 sd_inpaint=$D/SD_inpaint openforensics=$D/OpenForensics \
      --hint --hint-style v3 --out-dir $R/eval_v7 >> $L/eval_v7.log 2>&1 || { say "evaluation failed"; exit 1; }
fi
say "PIPELINE DONE"
