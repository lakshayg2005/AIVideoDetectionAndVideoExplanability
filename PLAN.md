# Plan: correct heatmaps and explanations across forgery types (images first)

Base paper: M2F2-Det (CVPR 2025). Goal: for any image, a real/fake verdict, a **correct** heatmap
(where the manipulation is), the **kind** of manipulation, and an explanation that names both.
Video and broader AI-generated content come after this works on images.

## Status (see RESULTS.md)
- Reproduced in 4-bit on one RTX 3060 (judgment acc 88.1%).
- Detector maps are faithful (mean-fill occlusion, p = 1.2e-7) but match human part labels only at
  chance: they show what the model uses, not yet where the forgery is.
- DDVQA explanation scores depend on the label and parts given in the question; stage 3 is being
  retrained on a self-explaining mix (`scripts/build_self_explain_data.py`). That model is the
  baseline for everything below.

## Forgery types and data (phase 1, final)
| # | Type | Data | Masks | Explanations |
|---|---|---|---|---|
| 1 | Face swap / reenactment (FF++) | DDVQA c40 + c23 (in repo) | weak: annotated part names | human |
| 2 | GAN face swap in the wild, multi-face | OpenForensics (ICCV 2021, CC-BY-4.0) | per face | generated from mask + face parts |
| 3 | Diffusion face swap | DiffusionFace DiffSwap (CC-BY-4.0) | face region | generated |
| 4 | Diffusion local edit / inpainting, faces | DiffusionFace Inpaint + own SD-inpainting of face parts | yes (own: exact) | generated |
| 5 | Diffusion img2img (re-rendered face) | DiffusionFace SDv1.5 img2img | whole image | generated |
| 6 | Fully generated faces | DiffusionFace ADM, SD1.5 text2img | whole image | generated |
| 7 | Photoshop splice / copy-move | CASIA v2 + revised GT masks | yes | MMTD-Set (GPT-4o) |
| 8 | Diffusion inpainting, general scenes | SD_inpaint + COCO real (FakeShield) | yes | MMTD-Set (GPT-4o) |
| 9 | Heavy filters / post-processing | own generator (`scripts/make_filtered_data.py`) on real and fake | edit mask | generated |

Held out for "unseen technique" tests: DiffusionFace LDM and SD2.1 text2img, DiffusionFace Wild,
OpenForensics Test-Dev, DDVQA c23 (unseen compression). Not covered: GAN attribute editing
(FaceApp / StarGAN; DFFD needs an instructor request) - partly approximated by type 4.
Optional later: FakeClue (FakeVLM) for human-style texts on AI-generated images.

## Architecture changes (on top of M2F2-Det)
M2F2-Det today: CLIP ViT-L/14-336 (LLaVA's tower, layer -2) -> (a) LLM image tokens, (b) detector:
576 patch-vs-forgery-prompt scores + CLIP CLS + DenseNet-121 -> Linear -> 2 probabilities ->
`deepfake_projector` -> one `<deepfake>` token for the LLM.

1. **Forgery mask head (heatmap).** FPN-style decoder fusing CLIP patch tokens (several layers,
   24x24), the 24x24 forgery-score map and DenseNet features (84x84 / 42x42 / 21x21) -> 2 channels at
   84x84, upsampled to the image: *forgery mask* (content manipulated) and *processing mask*
   (filters). Loss: BCE + Dice; the paper's 24x24 score map gets an auxiliary loss against the
   downsampled forgery mask. Real images: empty masks; fully generated: whole image; DDVQA: weak
   part-level supervision from the annotated parts (MediaPipe part masks).
2. **Edit-type head.** On [CLS, pooled DenseNet, mask-pooled patch features]: 6-way type
   (real, face swap/reenactment, splice/copy-move, diffusion local edit, img2img, fully generated)
   + 6 multi-label filters (beauty, blur, JPEG, colour, sharpen, resize). The original real/fake
   output is kept.
3. **Evidence and region tokens for the LLM.** `deepfake_projector` input grows from 2 to 14
   (real/fake + type + filters). In addition, CLIP patch features pooled inside the predicted mask
   (and inside the top face parts) go through the existing `mm_projector` as extra region tokens, so
   the LLM is told *what* and *where* before it explains.
4. **Inputs.** Face crops for face data (OpenForensics boxes, MediaPipe elsewhere); 336 px resize for
   scene images, sliding tiles for large photos in the mask head.

The CLIP vision tower stays frozen (it is shared with the LLM; changing it breaks the alignment and
the 12 GB budget).

## Training (all on the RTX 3060)
- **A. Detector + heads** (replaces the paper's stage 1): DenseNet, forgery prompts, projections,
  mask decoder, type head; CLIP frozen. Losses: real/fake CE + type CE + filter BCE + mask BCE/Dice +
  auxiliary map loss. Hours.
- **B. Projector alignment** (stage 2): 14-dim evidence + region tokens, 4-bit frozen LLM.
- **C. QLoRA explanations** (stage 3): DDVQA self-explaining + MMTD + generated grounded texts for
  OpenForensics, DiffusionFace, own edits and filters.

## Evaluation per type (and on held-out techniques)
Detection AUC; mask IoU / F1 and pointing game; edit-type and filter accuracy; explanation BLEU-4 /
METEOR / ROUGE-L / CIDEr / SPICE vs references; faithfulness (mean-fill occlusion); text-heatmap
consistency (parts / region named in the text vs predicted mask).

## Order
1. Finish self-explaining stage 3 and evaluate it (running).
2. Data index: images + masks + explanations for all sources; generated texts; filtered data.
3. Baseline evaluation of the current model on every type.
4. Training A, then B and C; re-evaluate; ablations (no mask loss, no region tokens, no type head).
5. Optional: RL for text-heatmap consistency with a smaller LLM; then video.
