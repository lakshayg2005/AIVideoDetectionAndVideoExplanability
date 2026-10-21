# Plan: correct heatmaps and explanations across forgery types (images first)

Base paper: M2F2-Det (CVPR 2025). Goal: for any image, a real/fake verdict, a **correct** heatmap
(where the manipulation is) and an explanation that names the evidence and the kind of edit.
Video and fully AI-generated content come after this works on images.

## Status (see RESULTS.md)
- Reproduced in 4-bit on one RTX 3060 (judgment acc 88.1%).
- Detector maps are faithful (mean-fill occlusion, p = 1.2e-7) but match human part labels only at
  chance: they show what the model uses, not yet where the forgery is.
- DDVQA explanation scores depend on the label and parts given in the question; stage 3 is being
  retrained on a self-explaining mix (`scripts/build_self_explain_data.py`).

## Architecture changes (on top of M2F2-Det)
1. **Mask head.** Light decoder on CLIP patch features + DenseNet features -> full-resolution forgery
   mask, Dice + BCE loss against ground-truth masks; the 24x24 CLIP forgery-score map is supervised
   with the downsampled mask. The heatmap becomes a trained prediction instead of a post-hoc readout.
2. **Edit-type head.** Multi-class manipulation type (real, face swap / reenactment, attribute edit,
   splice / copy-move, diffusion inpainting, fully generated) + multi-label filters (beauty smoothing,
   blur, JPEG, colour grading, sharpening, resize). The probabilities replace the 2-dim real/fake input
   of `deepfake_projector`, so the LLM knows which edit it is explaining.
3. **Region-aware explanations.** Stage-3 LoRA on texts that describe the tampered area and the edit type
   (MMTD-Set descriptions, DDVQA, generated filter descriptions), in the self-explaining format.
4. **High-resolution inputs.** Tile / crop large photos so small edits are not lost at 336 px.

## Data
| Data | Content | Masks | Explanations | Access |
|---|---|---|---|---|
| MMTD-Set-34k (FakeShield, ICLR 2025) | descriptions for the three sources below | via sources | yes (tampered area + judgment basis) | Hugging Face, Apache-2.0, downloaded |
| SD_inpaint | Stable-Diffusion inpainting + COCO real | yes | via MMTD | Hugging Face, Apache-2.0, 11.3 GB |
| CASIA v2 | Photoshop splice / copy-move | yes | via MMTD | Kaggle (account needed) |
| DFFD | FaceApp edits + FFHQ real | yes | via MMTD | MSU (request) |
| DDVQA c40/c23 | FF++ face swaps / reenactment | part names | human | in repo |
| Generated filters | beauty smoothing (skin mask), blur, JPEG, colour, sharpen on real and fake | yes | generated | local |
| FakeClue (FakeVLM, NeurIPS 2025) | faces + AI-generated | no | yes | Hugging Face, Apache-2.0, 29.8 GB |
| DiffusionFace | diffusion face synthesis / inpaint / swap | to check | no | GitHub |

Fully generated faces have no local region; for them the heatmap is evaluated only for faithfulness.

## Evaluation per category
Classic deepfake, attribute edit, splice, diffusion inpainting, fully generated, filtered real,
filtered fake: detection AUC; heatmap IoU / F1 and pointing game vs masks; edit-type accuracy;
explanation BLEU-4 / METEOR / ROUGE-L / CIDEr / SPICE vs references; faithfulness (mean-fill occlusion);
text-heatmap consistency.

## Order
1. Finish self-explaining stage 3 and evaluate it (running).
2. Data: download + organise mask/explanation sources; generate filtered data.
3. Evaluation suite on all categories with the current model (baseline).
4. Train detector with mask head + edit-type head (hours on the 3060).
5. Stage-2/3 with the K-dim edit-type input and region-aware texts (~1-2 days).
6. Re-evaluate; optional RL for text-heatmap consistency with a smaller LLM.
