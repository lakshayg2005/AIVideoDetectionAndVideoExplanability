# M2F2-Det Project Continuity & Session History Log

> **Purpose**: This file serves as a persistent, single-source-of-truth fallback across SSH reconnections and IDE session restarts. If a new chat session is started due to an SSH reconnect or server reload, any assistant should read this document first to immediately resume work without repeating past steps.

---

## System & Environment Specifications

- **Machine Hardware**:
  - GPU: NVIDIA GeForce RTX 3060 (12 GB VRAM)
  - System RAM: 16 GB
  - Storage: `/home/user2` (~635 GB available)
- **Target Repository & Paper**:
  - Paper: `2503.20188v1.pdf` (*M2F2-Det: Multi-Modal and Fine-Grained Multitask Deepfake Detection and Explanation*)
  - Workspace: `/home/user2/M2F2_Det`
- **Active Conda Environment**:
  - Path: `/home/user2/miniforge3/envs/M2F2_det`
  - Python: `3.10`
  - PyTorch: `2.1.2+cu121` (CUDA available: `True`, device: `NVIDIA GeForce RTX 3060`)
  - Key Libraries:
    - `numpy == 1.26.4` (Strictly pinned to prevent NumPy 2.x C-API breakage)
    - `transformers == 4.37.0`
    - `tokenizers == 0.15.1`
    - `bitsandbytes == 0.42.0` (Required for 4-bit loading on 12 GB VRAM)
    - `timm == 0.4.12`
    - `einops == 0.6.1`
    - `einops-exts == 0.0.4`
    - `gradio == 4.16.0`
    - `fastapi == 0.110.1`
    - `uvicorn`

---

## Chronological Session Log

### Session 1: October 12, 2026 (~21:47 - 00:30 IST)
- **Conversation ID**: `09c96996-0b27-4c59-b376-a66fba052036`
- **Database**: `~/.gemini/antigravity-ide/conversations/09c96996-0b27-4c59-b376-a66fba052036.db`
- **Actions Completed**:
  1. Assessed hardware feasibility for running M2F2-Det on an RTX 3060 12GB VRAM using `--load-4bit`.
  2. Installed Miniforge at `/home/user2/miniforge3`.
  3. Created dedicated conda environment `M2F2_det` with Python 3.10.
  4. Installed PyTorch with CUDA 12.1 support and primary vision/modeling libraries.
  5. Tested huggingface-cli. Encountered intermittent connection resets during package resolution right before the SSH connection dropped.

---

### Session 2: October 13, 2026 (~00:40 - 01:04 IST)
- **Conversation ID**: `16fe2689-f710-4e66-983c-a22731184ac2`
- **Database**: `~/.gemini/antigravity-ide/conversations/16fe2689-f710-4e66-983c-a22731184ac2.db`
- **Actions Completed**:
  1. Identified why previous chats disappear from the UI history dropdown after SSH reconnects (the remote IDE creates a new isolated extension host process and workspace storage directory, leaving the prior session detached).
  2. Fixed NumPy version incompatibility: downgraded `numpy` from 2.2.6 back to `1.26.4` to restore PyTorch C-extension compatibility.
  3. Installed web and serving libraries: `gradio==4.16.0`, `fastapi==0.110.1`, and `uvicorn`.
  4. Created a resilient Python download script with retries to download all 14 GB checkpoint weights for `CHELSEA234/llava-v1.5-7b-M2F2-Det` into `checkpoints/llava-v1.5-7b-M2F2-Det/`.
  5. Download verified:
     - `config.json` (1.9 KB)
     - `generation_config.json` (170 B)
     - `special_tokens_map.json` (552 B)
     - `tokenizer_config.json` (936 B)
     - `tokenizer.model` (489 KB)
     - `model.safetensors.index.json` (182 KB)
     - `model-00001-of-00003.safetensors` (4.6 GB)
     - `model-00002-of-00003.safetensors` (4.7 GB)
     - `model-00003-of-00003.safetensors` (4.3 GB)
  6. Inspected `config.json` and discovered the model architecture expects:
     - DenseNet deepfake encoder weights (`M2F2_Det_densenet121.pth`)
     - CLIP vision tower (`openai/clip-vit-large-patch14-336` or `vision_tower.pth`)
  7. Session disconnected before running inference tests.

---

### Session 3: October 13, 2026 (~11:27 IST - Present)
- **Conversation ID**: `33fc40a4-3dae-4b65-bdad-5ba10a3fbf61`
- **Database**: `~/.gemini/antigravity-ide/conversations/33fc40a4-3dae-4b65-bdad-5ba10a3fbf61.db`
- **Actions Completed**:
  1. Recovered context across server restarts; verified all 14 GB base weights, DenseNet detector (`1.99 GB`), and CLIP vision tower (`580 MB`).
  2. Created persistent continuity log `SESSION_HISTORY.md` in workspace root.
  3. Patched `llava/model/builder.py` and `llava/serve/cli_DDVQA_det.py` for 4-bit inference (`BitsAndBytesConfig`).
  4. Executed `stage_3_inference_det.sh` in 4-bit mode on RTX 3060 (fits comfortably in ~7.3 GB peak VRAM).
  5. **Diagnosed `<unk>` Output Degeneration**:
     - Lines 145+ in `outputs/DDVQA/DDVQA_det_c40.jsonl` produced repeated `<s><unk><unk><unk>...`.
     - Lines 1–144 were pre-existing valid samples (`The image is fake.`, `The image is real.`).
     - Line 145 was the first sample produced by the initial 4-bit run.
     - Isolated single-sample test on `0_384_932.jpg` revealed all 32,000 logits were identically `0.0`.
  6. **Root Cause Analysis & Complete Resolution**:
     - **Factor 1 (`safe_dispatch` parameter zeroing)**: An earlier `safe_dispatch` patch coupled with `low_cpu_mem_usage=False` was intercepting all 7 billion LLaMA parameters on the `meta` device and zeroing them out. Replaced with native `low_cpu_mem_usage=True` and `device_map='auto'`.
     - **Factor 2 (`llm_int8_skip_modules`)**: Added `'lm_head'`, `'mm_projector'`, `'deepfake_projector'`, `'deepfake_encoder'`, and `'vision_tower'` to skip 4-bit quantization, preserving them in native 16-bit float (`norm=198.0` for `lm_head`, `409.0` for projector).
     - **Factor 3 (BFloat16 vs Half mismatch)**: Fixed `deepfake_encoder` in `llava_llama.py` and `text_encoder.py` to use `torch.float16`, eliminating the `RuntimeError: expected scalar type BFloat16 but found Half`.
     - **Factor 4 (Single-Sample Verification)**: Ran `test_one_sample.py` on `0_384_932.jpg`. Generated valid logits (Step 0: `argmax=450 ('The')`, max logit `26.156`) and produced the exact prediction: `<s> The image is fake.</s>`.
  7. **Cleanup & Resume Ready**:
     - Truncated corrupt lines 145–209 from `outputs/DDVQA/DDVQA_det_c40.jsonl` (preserved original 144 valid predictions).
     - Added automatic resume support in `cli_DDVQA_det.py` to skip already processed images and resume directly from sample 145 onwards.
     - Completed full detection inference (`stage_3_inference_det.sh` finished with 144/144 valid predictions).

---

## Session 4: Explanation Inference Optimization & Live Execution
- **Date**: 2026-10-13
- **Goals**: Launch Stage 3 Explanation Inference (`stage_3_inference_exp.sh`), resolve environment activation and script configuration bugs, adapt script for 4-bit quantization, and verify output quality.
- **Key Actions & Fixes**:
  1. **Subshell Conda Activation & GPU Configuration**:
     - Fixed `CondaError: Run 'conda init' before 'conda activate'` in [`stage_3_inference_exp.sh`](file:///home/user2/M2F2_Det/stage_3_inference_exp.sh) by replacing `conda activate M2F2_det` with `source /home/user2/miniforge3/bin/activate M2F2_det`.
     - Changed `CUDA_NUM=1` to `CUDA_NUM=0` for the single-GPU system.
     - Appended `--load-4bit` to enable 4-bit NF4 quantized inference.
  2. **`cli_DDVQA_exp.py` 4-Bit Guard & Deepfake Encoder Placement**:
     - Guarded whole-model device move (`if not (args.load_4bit or args.load_8bit): model = model.to('cuda')`) to prevent `ValueError: .to() is not supported for 4-bit quantized models`.
     - Placed `deepfake_encoder` in CUDA FP16: `model.deepfake_encoder.to('cuda', dtype=torch.float16)`.
     - Added key-based resume tracking (`processed_keys`) so interrupted runs seamlessly continue without reprocessing.
     - Enabled real-time file flushing (`f_out.flush()`).
  3. **Verification**:
     - Launched `stage_3_inference_exp.sh` in background.
     - Evaluated first 60+ samples: confirmed coherent, detailed fine-grained explanations (e.g., skin texture smoothness, facial symmetry, eye/eyebrow consistency).
     - GPU VRAM steady at ~7.3 GB / 12 GB on RTX 3060.

---

## Current Status & Execution

| Pipeline Stage | Script | Status | Output File |
| :--- | :--- | :--- | :--- |
| **Detection** | `bash stage_3_inference_det.sh` | **Completed (144/144, 100%)** | [`outputs/DDVQA/DDVQA_det_c40.jsonl`](file:///home/user2/M2F2_Det/outputs/DDVQA/DDVQA_det_c40.jsonl) |
| **Explanation** | `bash stage_3_inference_exp.sh` | **Completed (1070/1070, 100%)** | [`outputs/DDVQA/DDVQA_exp_c40.jsonl`](file:///home/user2/M2F2_Det/outputs/DDVQA/DDVQA_exp_c40.jsonl) |

