# BTP meeting brief: explainable face / image forgery detection (built on M2F2-Det)

## 1. One-paragraph summary

M2F2-Det (CVPR 2025 Oral) detects face forgeries and explains them with an LLM, but its explanations depend on the
facial parts given in the prompt, and its heatmaps are not trained on any ground-truth forged region. We added
mask-supervised forensic heads that give a heatmap of the forged region, the edit type (face swap, splice, diffusion
inpainting, re-rendered face, fully AI-generated) and the filters applied, and connected them to the LLM so that its
explanation names the edit type and refers to the region the heatmap marks. It now covers face swaps, diffusion faces,
diffusion edits, Photoshop-style splicing, AI inpainting of scenes and filtered images.

## 2. The base paper and what it uses

- Paper: **M2F2-Det**, CVPR 2025 Oral, arXiv 2503.20188. Released code and weights are in this repository.
- **Vision encoder:** CLIP ViT-L/14-336 (fine-tuned by the authors); layer -2 patch features.
- **Detector:** forgery-prompt CLIP text branch (learned prompt tokens), 576 patch cosine scores, DenseNet-121, and a
  linear output layer over [text score, CLIP feature, DenseNet feature]; output real / fake.
- **LLM:** LLaVA-1.5-7B. The detector's 2-dim output goes through a `deepfake_projector` (2 -> 4096) and becomes one
  `<deepfake>` token in the prompt.
- **Training data:** DDVQA, FF++ frames (c40 compression) with human-written explanations.
- Reported in the paper: accuracy 95.23 / F1 96.61. **Our reproduction with the released weights: 88.1 / 87.6**
  (the authors' own cached outputs score 76.4 / 71.2). Explanation scores were within 2-8% of the cached ones.

Findings that motivated our changes (RESULTS.md sections 1-2):
- The detector's evidence maps are faithful to the model (mean-fill test p = 1e-7) but score close to chance against
  human-marked face parts.
- DDVQA explanation scores are driven by the face parts written in the prompt: CIDEr falls from 0.743 to about 0.10
  when they are removed.
- On unseen diffusion faces the released model's verdict accuracy is 4% (it calls them real).

## 3. Architectural changes

| # | Change | Why |
|---|---|---|
| 1 | **Mask decoder** (FPN over CLIP layers 8 / 16 / -2, the 24x24 score map and DenseNet features) giving a forged-region mask and a processing (filter) mask | the paper has no pixel-level supervision; ours is trained on ground-truth masks |
| 2 | **Type head** (6 types) and **filter head** (beauty, blur, JPEG, colour, sharpen, resize) | the explanation must name the edit and the filters |
| 3 | **Distillation** from the frozen released detector | keeps its detection strength on FF++ while the masks are learned |
| 4 | **High-res branch (v5/v6)**: 672 px, SRM noise stem, log-FFT, frozen DINOv2-B, giving a 168x168 heatmap | scene edits are small; 336 px is too coarse |
| 5 | **LLM input**: 16-dim evidence token (verdict, type, filters) plus 3 region tokens (CLIP patches inside the predicted forged area, outside it, and in the filtered area) | the explainer sees where the heads point, not only a real / fake score |
| 6 | **Function-preserving projector init** from the released 2-dim projector | training from plain LLaVA collapsed; this starts from the released behaviour |
| 7 | **Stage B** (projector only, LLM frozen) then **Stage C** (QLoRA r=128 on the LLM) | cheap alignment first, then the language |
| 8 | **Hybrid UI**: the heads decide; the explainer explains with a hint naming the type and the marked region; three-signal uncertainty banner and text checks | the heads are more accurate than the LLM's own verdict |

Training texts (Stage C) were rebuilt: a teacher VLM names the exact edited region, lighting / shadow boilerplate
(93.5% of MMTD, 99.9% of FakeClue answers) is cut from 85% of texts, and the prompt carries the heatmap hint.

## 4. Datasets and what each is used for

| Dataset | Content | Used for |
|---|---|---|
| **DDVQA c40** (in repo) | FF++ face-swap frames, human explanations | reproduction, face-swap evaluation, Stage C explanation data |
| **OpenForensics** (Zenodo, CC-BY) | faces with pixel masks; Train / Val / Test-Dev / Test-Challenge | heatmap supervision and IoU evaluation |
| **DiffusionFace** (Zenodo, CC-BY) | diffusion-generated and inpainted faces; Inpaint subset has masks and edit prompts | diffusion faces and edits; LDM and SD-2.1 held out as unseen generators |
| **CASIA v2** + ground-truth masks | Photoshop splice / copy-move | splicing heatmaps, type, evaluation |
| **SD_inpaint** (HF) | scene images with diffusion-inpainted regions and masks | scene inpainting heatmaps |
| **MMTD-Set-34k** | GPT-4o-written explanations for CASIA and SD_inpaint images | Stage C explanation texts |
| **FakeClue** (FakeVLM, Apache-2.0) | FF++ frames and AI-generated people with descriptions | type head, AI-generated people, Stage C texts |
| **Teacher texts** (ours) | 2,800 region names from Qwen2.5-VL-3B shown the ground-truth box | exact region names ("the head of the panda") |
| **Synthetic filters** (ours, on the fly) | beauty, blur, JPEG, colour, sharpen, resize | filter head and robustness |

Dropped: DFFD (access needs an instructor request).

## 5. Models and why

| Model | Role | Why |
|---|---|---|
| LLaVA-1.5-7B | explainer LLM | the paper's own; released weights; fits a 12 GB GPU in 4-bit QLoRA |
| CLIP ViT-L/14-336 | vision backbone | the paper's own; detector and LLM share it |
| DenseNet-121 | detector branch | the paper's own |
| DINOv2-B | high-res branch (v6) | better localisation of scene edits (SD_inpaint IoU 0.403 against 0.314 at equal steps) but 3x slower |
| MediaPipe FaceMesh | face parts; confirms a human face | the Haar detector fired on cats and objects |
| Qwen2.5-VL-3B | offline teacher for region names only | newer VLM; not part of the final system |
| bitsandbytes 4-bit NF4 | memory | 7B LLM on an RTX 3060 (12 GB) |

Not done: swapping LLaVA for another LLM or CLIP for a stronger encoder. It is listed as a future ablation.

## 6. Results

### Heads (Stage A), final checkpoints

| Test set | v3 (in the LLM) | v6 (best on scenes) |
|---|---|---|
| OpenForensics Test-Dev IoU | **0.914** | 0.876 |
| OpenForensics Test-Challenge IoU | **0.753** | 0.500 |
| SD_inpaint IoU | 0.406 | **0.483** |
| CASIA IoU | 0.312 | 0.286 |
| DiffusionFace edit IoU | not supervised | 0.296 (v5 best: 0.443) |
| DDVQA (FF++) AUC | **0.865** | 0.848 |
| SD_inpaint AUC | 0.947 | **0.954** |
| AI-generated people AUC (FakeClue) | not trained | **0.990** |
| Unseen-generator TPR | 0.937 | 0.930 |

### Explanations (648 test images, the UI's open question with the heatmap hint)

Verdict accuracy (DDVQA / DiffusionFace / unseen generators / CASIA / SD_inpaint):

| | DDVQA | DiffFace | Unseen gen. | CASIA | SD_inpaint |
|---|---|---|---|---|---|
| Released M2F2-Det (no hint) | 0.87 | 0.58 | 0.04 | 0.52 | 0.49 |
| **Stage C v1** (earlier UI) | **0.81** | 0.87 | 0.69 | 0.81 | **0.70** |
| **Stage C v2** (UI rule) | 0.72 | **0.97** | **0.97** | **0.89** | 0.69 |
| **Stage C v3** (newest) | 0.70 | **0.98** | 0.96 | 0.88 | **0.84** |

Other measures (CASIA / SD_inpaint):

| | Answers mentioning lighting | Position named correctly, when named |
|---|---|---|
| Released | 7% / 1% | 0.40 / 0.25 |
| v1 | **95% / 97%** | 0.57 / 0.76 |
| v2 | **8% / 0%** | 0.57 / 0.75 |
| v3 | 10% / 4% | **0.78 / 0.80** |

Answer correctly names the edit type (fakes): v1 0.60 / 0.47 / 0.68 / 0.51 / 0.00; v2 0.78 / 0.61 / 0.94 / 0.08 / 0.90
(same column order). v2's weak spot: CASIA splices are called "inpainting". v3 is trained to fix this.

## 7. Known limitations (say these first)

- Compressed FF++ **real** faces: the heads call 56% fake, and the explainer follows them (DDVQA accuracy 0.72).
- CASIA heatmaps are modest (IoU 0.29-0.35); only the OpenForensics faces reach 0.9.
- The explainer follows the heads' verdict; if the heads are wrong, the explanation justifies the wrong verdict.
- The 15 demo images are held out, but they are a small sample, not a benchmark.
- Reproduction is 88.1 against 95.23 in the paper (released weights, our evaluation script).

## 8. Running the UI in front of the professor

Time needed: the model loads in about 3-4 minutes, so start it 15 minutes before.

```
cd /home/cse/Desktop/Lakshay/BTP-1/AIVideoDetectionAndVideoExplanability
bash run_forensic_ui.sh --stage-c3        # or --stage-c2 (see below)
```

Wait for the line `Ready.`, then open **http://127.0.0.1:7861** in the browser. Run one image first to warm it up (the
first answer is slower).

- `--stage-c2` / `--stage-c3`: explainer version (v2 / v3). Use the one chosen after tonight's test.
- `--no-llm`: heatmaps, verdict and type only; runs on the CPU, starts in under a minute. **Fallback if the GPU
  misbehaves.** Tested working on 28 Oct.
- `--share`: also makes a public link; needs internet, not needed for a demo on this machine.
- Only one GPU program at a time: nothing else (training, evaluation) may run during the demo.
- Check the GPU is free first: `nvidia-smi` should show about 100 MiB used.

Suggested order (images in `../demo_images/`; confirm after tonight's test run):
1. `REAL_photo_1.jpg`: real photo, no manipulation.
2. `FAKE_diffusion_face_edit_1.png`: diffusion face edit; heatmap on the edited part.
3. `FAKE_photoshop_splice_1.jpg`: splice; heatmap and the region named.
4. `FAKE_AI_inpainting_1.jpg`: scene inpainting.
5. `FAKE_AIgenerated_LDM_unseen_1.png`: unseen generator; no single region.
6. Mention `REAL_FF++_1/2` as the known failure (section 7) rather than hoping it passes.
