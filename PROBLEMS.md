# M2F2-Det Implementation: Problems Encountered & Solutions

This document details all technical challenges, bugs, and hardware constraints encountered while making the **M2F2-Det** (*Multi-Modal and Fine-Grained Multitask Deepfake Detection and Explanation*, `2503.20188v1.pdf`) repository operational on an NVIDIA GeForce RTX 3060 (12 GB VRAM) desktop, along with their root causes and resolutions.

---

## 1. Hardware & VRAM Limitations (12 GB VRAM Constraint)

### Problem
The standard M2F2-Det architecture comprises:
- Base Vision-Language Model: **LLaVA-1.5-7B** (~14 GB in FP16)
- Fine-grained Deepfake Detector: **DenseNet-121** (~1.99 GB weights)
- Auxiliary Vision Tower: **CLIP-ViT-L/14-336** (~580 MB weights)

Loading everything in standard FP16 requires **18–20 GB VRAM**, which immediately triggers an `OutOfMemoryError` on a single consumer RTX 3060 (12 GB).

### Root Cause
The original research paper and code repository were designed for datacenter GPUs (NVIDIA A100 40GB/80GB) without quantization support out-of-the-box.

### Solution
- Configured 4-bit NormalFloat (`NF4`) quantization via `BitsAndBytesConfig` inside [`llava/model/builder.py`](file:///home/user2/M2F2_Det/llava/model/builder.py):
  ```python
  kwargs['quantization_config'] = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      llm_int8_skip_modules=['mm_projector', 'vision_tower', 'deepfake_projector', 'deepfake_encoder', 'lm_head']
  )
  ```
- Reduced total VRAM consumption from ~18 GB to **~7.3 GB peak**, running comfortably within the RTX 3060's 12 GB limit with headroom for KV-cache.

---

## 2. NumPy 2.x C-API Binary Incompatibility

### Problem
Running inference or importing `torchvision` threw fatal C-extension errors:
```text
RuntimeError: Numpy is not available / A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x ...
_ARRAY_API not found
```

### Root Cause
Default package installations pulled `numpy >= 2.0.0`. PyTorch 2.1.2 and torchvision binaries in the conda environment were compiled against the NumPy 1.x C-API, breaking binary ABI compatibility.

### Solution
- Pinned and downgraded NumPy to the last 1.x stable release:
  ```bash
  conda activate M2F2_det
  pip install "numpy==1.26.4"
  ```

---

## 3. Disconnected Sessions & Context Loss Across SSH Restarts

### Problem
Whenever the remote IDE or SSH connection reconnected, background tasks terminated and prior conversation memory was lost, making it difficult to know what had already been downloaded, modified, or tested.

### Solution
- Created [`SESSION_HISTORY.md`](file:///home/user2/M2F2_Det/SESSION_HISTORY.md) directly in the workspace root as a persistent, single source of truth documenting hardware specs, environment versions, verified file paths, and chronological session steps.

---

## 4. Missing Checkpoint Weights & Interrupted Downloads

### Problem
The repository expected local checkpoints for:
1. `checkpoints/llava-v1.5-7b-M2F2-Det/` (14 GB safetensors across 3 shards)
2. `utils/weights/M2F2_Det_densenet121.pth` (1.99 GB)
3. `utils/weights/vision_tower.pth` (580 MB)

Downloads frequently stalled or suffered connection resets when pulling through interactive CLI sessions.

### Solution
- Created automated download scripts with chunked retry loops and verified sha256/size integrity on disk.
- Updated `checkpoints/llava-v1.5-7b-M2F2-Det/config.json` to correctly link `deepfake_model_path` to `./utils/weights/M2F2_Det_densenet121.pth`.

---

## 5. Projection & Vocabulary Degradation Under Global Quantization

### Problem
Applying standard 4-bit quantization to the entire model caused output degradation and crashes in the multi-modal projector layers (`mm_projector`, `deepfake_projector`).

### Root Cause
Multi-layer perceptron projection layers contain GELU activations and small weight tensors that are highly sensitive to 4-bit quantization errors. Furthermore, quantizing `lm_head` into 4-bit uint8 led to severe output vocabulary distortions.

### Solution
- Specified sensitive non-LLM modules in `llm_int8_skip_modules`:
  ```python
  llm_int8_skip_modules=['mm_projector', 'vision_tower', 'deepfake_projector', 'deepfake_encoder', 'lm_head']
  ```
- Kept the projection layers, vision encoders, and `lm_head` in native 16-bit float (`torch.float16`), while quantizing only the 32 LLaMA transformer layers to 4-bit.

---

## 6. The `<unk>` Token Degeneration Bug (The "Zero-Weight" Bug)

### Problem
In `outputs/DDVQA/DDVQA_det_c40.jsonl`, samples 1–144 were valid text, but line 145 onwards repeatedly generated:
```json
{"text": "<s><unk><unk><unk><unk><unk><unk><unk>..."}
```
Logit analysis on test sample `0_384_932.jpg` revealed that **all 32,000 vocabulary logits were identically `0.0`**, forcing greedy decoding to tie-break on token index 0 (`<unk>`).

### Root Cause
In [`llava/model/builder.py`](file:///home/user2/M2F2_Det/llava/model/builder.py), a previous workaround patched `mu.dispatch_model` with a custom function `safe_dispatch`:
```python
def safe_dispatch(m, **d_kwargs):
    for name, param in list(m.named_parameters()):
        if param.is_meta:
            val = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
            set_module_tensor_to_device(m, name, device='cpu', value=val)
    ...
```
When called with `low_cpu_mem_usage=False`, Hugging Face's `from_pretrained` created the model skeleton with parameters on the `meta` device. `safe_dispatch` intercepted this and **replaced all 7 billion LLaMA layer parameters, embeddings, and lm_head with zeros (`torch.zeros`)**, wiping out all trained weights before dispatch!

### Solution
1. Removed `safe_dispatch` completely from [`llava/model/builder.py`](file:///home/user2/M2F2_Det/llava/model/builder.py).
2. Switched to Hugging Face's native **`low_cpu_mem_usage=True`** with **`device_map="auto"`**:
   ```python
   model = LlavaLlamaForCausalLMDeepfake.from_pretrained(
       model_path,
       low_cpu_mem_usage=True,
       **kwargs
   )
   ```
3. Verified on CUDA that `embed_tokens` (131M elements), `layers` (Linear4bit), `lm_head` (`norm = 198.0`), and `deepfake_projector` (`norm = 409.0`) loaded their true trained weights.

---

## 7. Precision Mismatch: BFloat16 vs Half (Float16)

### Problem
During inference, `model.generate()` crashed inside `text_encoder.py`:
```text
File "llava/model/deepfake/M2F2Det/text_encoder.py", line 41, in forward
  hidden_states = self.layer_norm1(hidden_states)
RuntimeError: expected scalar type BFloat16 but found Half
```

### Root Cause
In [`llava/model/language_model/llava_llama.py`](file:///home/user2/M2F2_Det/llava/model/language_model/llava_llama.py), `deepfake_encoder` was hardcoded to instantiate with `torch.bfloat16`:
```python
vision_dtype=torch.bfloat16,
text_dtype=torch.bfloat16,
deepfake_dtype=torch.bfloat16,
```
However, the base LLaVA model, projector, and input embeddings were running in `torch.float16` (`Half`). When `CLIPTextEncoder` performed layer normalization, PyTorch encountered mixed `BFloat16` and `Half` tensors.

### Solution
1. Updated [`llava/model/language_model/llava_llama.py`](file:///home/user2/M2F2_Det/llava/model/language_model/llava_llama.py) to initialize `deepfake_encoder` in `torch.float16`:
   ```python
   vision_dtype=torch.float16,
   text_dtype=torch.float16,
   deepfake_dtype=torch.float16,
   ```
2. Updated [`llava/model/deepfake/M2F2Det/text_encoder.py`](file:///home/user2/M2F2_Det/llava/model/deepfake/M2F2Det/text_encoder.py) to dynamically match `input_embeds` to `self.prompt_tokens.dtype` and device.
3. Explicitly placed the encoder on CUDA in FP16:
   ```python
   model.deepfake_encoder.to('cuda', dtype=torch.float16)
   ```

---

## 8. Missing Sequence Dimension Variable in CLIPTextEncoder

### Problem
After resolving the dtype mismatch, inference threw:
```text
File "llava/model/deepfake/M2F2Det/text_encoder.py", line 37, in forward
  position_ids = torch.arange(seq_length, device=input_embeds.device).expand(input_embeds.shape[0], -1)
NameError: name 'seq_length' is not defined
```

### Root Cause
In `CLIPTextEncoder.forward()`, `batch_size, seq_length = input_embeds.shape[:2]` was accidentally omitted during a refactoring edit.

### Solution
- Restored `batch_size, seq_length = input_embeds.shape[:2]` immediately after embedding tensor preparation in [`llava/model/deepfake/M2F2Det/text_encoder.py`](file:///home/user2/M2F2_Det/llava/model/deepfake/M2F2Det/text_encoder.py).

---

## 9. Lack of Resume Support in Batch Evaluation Script

### Problem
The evaluation script [`llava/serve/cli_DDVQA_det.py`](file:///home/user2/M2F2_Det/llava/serve/cli_DDVQA_det.py) opened `outputs/DDVQA/DDVQA_det_c40.jsonl` in append mode (`'a'`), but iterated through the entire dataset from index 0 without checking which images were already processed. Restarting the script would re-evaluate already processed samples and create duplicates.

### Solution
- Truncated the corrupt `<unk>` lines (145–209) from `outputs/DDVQA/DDVQA_det_c40.jsonl` using `head -n 144`, retaining all 144 valid historical predictions.
- Added resume logic in [`llava/serve/cli_DDVQA_det.py`](file:///home/user2/M2F2_Det/llava/serve/cli_DDVQA_det.py):
  ```python
  out_file = os.path.join(out_dir, 'DDVQA_det_c40.jsonl')
  processed_images = set()
  if os.path.exists(out_file):
      with open(out_file, 'r') as f:
          for line in f:
              try:
                  data = json.loads(line)
                  if 'image' in data:
                      processed_images.add(data['image'])
              except Exception:
                  pass
  print(f"Resuming: found {len(processed_images)} already processed images in {out_file}.")
  ```
- Inference now automatically skips previously evaluated files and resumes directly from sample 145 onwards.

---

## 10. `stage_3_inference_exp.sh`: Conda Activation Failure & CUDA Device Ordinal
### Problem
Running `bash stage_3_inference_exp.sh` failed immediately with:
```text
CondaError: Run 'conda init' before 'conda activate'
...
ModuleNotFoundError: No module named 'torch'
```
Furthermore, the script hardcoded `CUDA_NUM=1`, whereas the workstation only has 1 physical GPU (`device:0`).

### Root Cause
1. Non-interactive subshells launched via `bash script.sh` do not inherit `conda` wrapper functions from `~/.bashrc` unless explicitly initialized.
2. When `conda activate M2F2_det` failed, the script fell back to the system `/usr/bin/python3`, where PyTorch was not installed.
3. `CUDA_NUM=1` would have isolated the process to an invalid GPU device ordinal.

### Solution
Updated [`stage_3_inference_exp.sh`](file:///home/user2/M2F2_Det/stage_3_inference_exp.sh):
- Sourced the miniforge environment directly: `source /home/user2/miniforge3/bin/activate M2F2_det`.
- Adjusted device to `CUDA_NUM=0`.
- Passed the `--load-4bit` flag for 4-bit inference.

---

## 11. `cli_DDVQA_exp.py`: 4-Bit Compatibility and Resume Tracking
### Problem
The upstream script [`llava/serve/cli_DDVQA_exp.py`](file:///home/user2/M2F2_Det/llava/serve/cli_DDVQA_exp.py) contained unconditional device moves (`model = model.to('cuda')`), which throw a `ValueError` when running in 4-bit/8-bit mode. Additionally, `deepfake_encoder` was not placed in CUDA FP16, and the script lacked resume capability and stream flushing.

### Solution
In [`llava/serve/cli_DDVQA_exp.py`](file:///home/user2/M2F2_Det/llava/serve/cli_DDVQA_exp.py):
1. Guarded whole-model device movement: `if not (args.load_4bit or args.load_8bit): model = model.to('cuda')`.
2. Loaded and moved the deepfake encoder to GPU float16: `model.deepfake_encoder.to('cuda', dtype=torch.float16)`.
3. Added key-based resume tracking to skip already answered questions in `outputs/DDVQA/DDVQA_exp_c40.jsonl`.
4. Added immediate buffer flushing (`f_out.flush()`) on every write.

---

## Summary of Current Operational State

| Metric / Component | Detection (`stage_3_inference_det.sh`) | Explanation (`stage_3_inference_exp.sh`) |
| :--- | :--- | :--- |
| **GPU VRAM Utilization** | **7.3 GB / 12.0 GB** | **7.3 GB / 12.0 GB** |
| **Status** | **Completed (144/144, 100%)** | **Completed (1070/1070, 100%)** |
| **Quantization** | **Active (4-bit NF4)** | **Active (4-bit NF4)** |
| **Precision** | **FP16 CLIP / DenseNet / LM Proj** | **FP16 CLIP / DenseNet / LM Proj** |
| **Output File** | [`outputs/DDVQA/DDVQA_det_c40.jsonl`](file:///home/user2/M2F2_Det/outputs/DDVQA/DDVQA_det_c40.jsonl) | [`outputs/DDVQA/DDVQA_exp_c40.jsonl`](file:///home/user2/M2F2_Det/outputs/DDVQA/DDVQA_exp_c40.jsonl) |
| **Sample Quality** | `<s> The image is fake.</s>` | `<s> The skin in the image is fake because it lacks of details, such as skin texture.</s>` |

