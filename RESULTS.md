# Results on a single RTX 3060 (12 GB)

All runs: DDVQA c40 test split shipped in this repo (144 face crops, 72 real / 72 fake),
released M2F2-Det weights, 4-bit NF4 LLM (vision tower, detector, projectors and `lm_head` kept in fp16).

## 0. Reproduction

Judgment (`eval/eval_judgement.py`):

| Run | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| Paper (Fig. 5a, full DD-VQA test) | 95.23 | 96.61 | – | – |
| Authors' cached `outputs/DDVQA/DDVQA_det_c40.jsonl` | 76.39 | 71.19 | 91.30 | 58.33 |
| **Ours, 4-bit** `outputs/repro/DDVQA_det_c40.jsonl` | **88.11** | **87.59** | 90.91 | 84.51 |

One of our 144 answers named neither "real" nor "fake" and is excluded by the script.

Explanation (`eval/eval_explanation.py`, 1070 questions; the paper only plots these in Fig. 5b):

| Run | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---|---|---|---|
| Authors' cached `outputs/DDVQA/DDVQA_exp_c40.jsonl` | 0.320 | 0.268 | 0.602 | 2.734 | 0.598 |
| **Ours, 4-bit** `outputs/repro/DDVQA_exp_c40.jsonl` | 0.294 | 0.280 | 0.590 | 2.632 | 0.574 |

## 1. Detector evidence maps (`explain/run_ddvqa_maps.py`)

- Detector head alone: acc@0.5 = 0.722, AUC = 0.863 (the LLM verdict is better calibrated).
- Part localization vs human-annotated parts (68 fakes): hit@1 fused 0.529 / patch 0.471 / Grad-CAM 0.515
  vs random 0.536. The maps do not pick the annotated part better than chance.
- Occlusion faithfulness, drop in fake-real logit margin (Wilcoxon signed-rank, n = 72 per class):

| Occlusion | class | top part | random part | bottom part | top > bottom | top > random |
|---|---|---|---|---|---|---|
| blur | fake | -1.10 | -1.47 | -1.42 | 56% (p = 0.24) | 56% (p = 0.30) |
| mean fill | fake | **+0.48** | -0.72 | -1.56 | **81% (p = 1.2e-7)** | **65% (p = 0.0013)** |
| mean fill | real | -1.68 | -2.70 | -3.01 | 69% (p = 3.6e-4) | 65% (p = 0.018) |

The map-ranked part is significantly more decision-relevant than other parts. Blurring any part makes
the image look *more* fake, so blur is a spurious forgery cue for the detector (and a confound for blur-based
faithfulness tests).

## 2. Self-grounded explanations (`explain/grounded_explain.py`)

The DDVQA protocol puts the ground-truth label and the annotated parts into the question. Here the verdict is
the model's own answer and the parts are the top-2 parts of the detector map.

- verdict accuracy 87.5%, text-map consistency 0.866 (share of parts mentioned in the answer that the map selected)

Overall question only (144 answers, scored against the human answers):

| Prompt inputs | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---|---|---|---|
| GT label + GT parts (DDVQA protocol, 144 answers) | 0.263 | 0.232 | 0.405 | 0.743 | 0.478 |
| GT label + GT parts via this pipeline (sanity check, 137) | 0.262 | 0.232 | 0.408 | 0.747 | 0.486 |
| model verdict + GT parts (137) | 0.244 | 0.223 | 0.387 | 0.667 | 0.467 |
| GT label + map parts (137) | 0.053 | 0.109 | 0.210 | 0.104 | 0.225 |
| GT label + random parts (137) | 0.089 | 0.120 | 0.214 | 0.106 | 0.242 |
| model verdict + map parts (self-grounded, 144) | 0.046 | 0.102 | 0.193 | 0.073 | 0.209 |

Ablation outputs: `outputs/explain_ablation/label-*_parts-*` (`explain/grounded_explain.py --overall-only
--label-source --parts-source`; 7 test images have no overall question and are skipped when GT inputs are used).

Without the oracle inputs, text-similarity scores collapse, and almost all of the drop comes from the part list:
with the true label kept, replacing the annotated parts by map parts or random parts gives the same low scores.
The DDVQA explanation metrics mostly reward describing the parts named in the question, not explaining the
model's own decision.

## 3. Stage-A pilot: mask-supervised detector (`forensics/`, PLAN.md)

Kaggle T4, 8000 steps x batch 12 (2.8 h), sources sampled evenly: CASIA v2 (splice / copy-move), SD_inpaint
(diffusion inpainting, scenes), DDVQA c40 (face swap, weak part masks), OpenForensics Val split 80/20 by image
(GAN face swap in the wild). 30% of training images get random filters. Validation: up to 300 images per
source and class, plus a fixed-seed filtered copy of the face sets. Log: `outputs/stagea_pilot/log.jsonl`.

Mask localisation on fake images with pixel masks. "Paper map" = the released detector's exact patch
attribution (section 1) before any training; "ours" = the new mask head after 8000 steps.

| Source | pointing game, paper map | pointing game, ours | mask IoU | mask F1 | false-positive area on reals |
|---|---|---|---|---|---|
| OpenForensics (face swap) | 0.870 | **0.997** | **0.919** | **0.957** | 0.003 |
| OpenForensics, filtered | 0.757 | **0.990** | 0.851 | 0.903 | 0.006 |
| SD_inpaint (scene inpainting) | 0.167 | **0.820** | 0.383 | 0.473 | 0.005 |
| CASIA v2 (splice / copy-move) | 0.113 | **0.573** | 0.279 | 0.352 | 0.005 |

Detection (AUC) and type / filter heads:

| Source | AUC at step 0 (released detector) | AUC ours | type acc | filter F1 | processing-mask IoU |
|---|---|---|---|---|---|
| OpenForensics | 0.752 | **0.994** | 0.995 | – | – |
| OpenForensics, filtered | 0.586 | **0.978** | 0.970 | 0.795 | 0.861 |
| CASIA v2 | 0.654 | **0.936** | 0.908 | – | – |
| SD_inpaint | 0.474 | **0.760** | 0.775 | – | – |
| DDVQA c40 | **0.863** | 0.811 | 0.722 | – | – |
| DDVQA c40, filtered | **0.851** | 0.779 | 0.729 | 0.790 | 0.918 |

- Faces with pixel masks are localised almost perfectly; scene edits (small regions resized to 336 px) much less.
- Masks do not fire on real images (false-positive area <= 0.6%), also after filtering.
- Regression: detection on DDVQA c40 (the paper's own FF++ domain) drops from 0.863 to 0.811 AUC: 1.3K
  heavily re-sampled training images next to much larger sources.
- Caveat: OpenForensics validation comes from the same split as training; the full run uses the official
  Train / Val / Test-Dev / Test-Challenge splits.
- After training, the paper map's pointing game is not comparable (the output layer weights change sign),
  so only its step-0 value is reported.

## 4. Stage-A v2: all data, official test splits (`kaggle/stagea_v2.py`)

> **Correction:** the v2 DiffusionFace index also listed the 30K Inpaint *mask* PNGs (black-and-white images) as fake
> samples, and Inpaint masks were not used. The DiffusionFace rows below are therefore inflated, and the Inpaint
> subset had no localisation supervision. Fixed in `forensics/index.py`; v3 (section 5) re-runs Stage A on the
> corrected index. The OpenForensics, SD_inpaint, CASIA and DDVQA rows are unaffected by the index error itself.

Kaggle T4, 20000 steps x batch 12 (6 h), 343K training images: CASIA v2, SD_inpaint, DDVQA c40, OpenForensics
(official Train; Val / Test-Dev / Test-Challenge never trained on), DiffusionFace (Real, ADM, SD1.5 text2img,
SD1.5 img2img, Inpaint, DiffSwap; **LDM and SD2.1 text2img held out** as unseen generators). Distillation from
the released detector on DDVQA. Up to 300 evaluation images per source and class, plus a fixed-seed filtered
copy of every face set. Log: `outputs/stagea_v2/log.jsonl`. Numbers below are at step 20000.

Localisation on fakes with pixel masks (paper map = released detector, step 0):

| Test set | pointing, paper map | pointing, ours | IoU | F1 | mask area on reals |
|---|---|---|---|---|---|
| OpenForensics Test-Dev | 0.843 | **1.000** | **0.914** | **0.954** | 0.7% |
| OpenForensics Test-Dev, filtered | 0.773 | **0.993** | 0.857 | 0.907 | 0.2% |
| OpenForensics Test-Challenge (perturbed, harder) | 0.720 | **0.957** | 0.726 | 0.777 | 2.0% |
| OpenForensics Test-Challenge, filtered | 0.700 | **0.970** | 0.705 | 0.765 | 2.1% |
| SD_inpaint (scenes) | 0.167 | **0.813** | 0.387 | 0.483 | 0.4% |
| CASIA v2 (scenes) | 0.113 | **0.593** | 0.327 | 0.407 | 0.3% |

Detection and type recognition (AUC step 0 -> 20000; FPR = real images called fake at 0.5):

| Test set | AUC | FPR | TPR | type acc | filter F1 |
|---|---|---|---|---|---|
| OpenForensics Test-Dev | 0.729 -> **0.998** | 0.047 | 0.990 | 0.997 | – |
| OpenForensics Test-Dev, filtered | 0.611 -> **0.991** | 0.033 | 0.943 | 0.982 | 0.831 |
| OpenForensics Test-Challenge | 0.497 -> **0.926** | 0.179 | 0.867 | 0.918 | – |
| OpenForensics Test-Challenge, filtered | 0.492 -> **0.906** | 0.207 | 0.867 | 0.900 | 0.735 |
| DiffusionFace (seen generators) | 0.718 -> **0.996** | 0.030 | 0.960 | 0.998 | – |
| DiffusionFace, filtered | 0.664 -> **0.938** | 0.133 | 0.857 | 0.892 | 0.865 |
| DiffusionFace **unseen generators** (fakes only) | – | – | 0.123 -> **0.937** | 0.980 | – |
| DiffusionFace unseen, filtered | – | – | 0.223 -> **0.800** | 0.793 | 0.845 |
| SD_inpaint | 0.474 -> **0.943** | 0.197 | 0.900 | 0.877 | – |
| CASIA v2 | 0.654 -> **0.944** | 0.120 | 0.863 | 0.905 | – |
| DDVQA c40 | **0.863** -> 0.845 | 0.569 | 0.931 | 0.764 | – |
| DDVQA c40, filtered | **0.851** -> 0.820 | 0.500 | 0.889 | 0.743 | 0.829 |

- Face swaps on held-out test images are localised almost perfectly (Test-Dev IoU 0.91, pointing 1.00), and
  still well on the perturbed Test-Challenge set (IoU 0.73, pointing 0.96).
- Unseen diffusion generators (LDM, SD 2.1) are detected 94% of the time (80% after filtering).
- Distillation reduced the DDVQA regression from -0.052 (pilot) to -0.018 AUC. Its high FPR at 0.5 is already
  present in the released detector (0.514 at step 0): the head is poorly calibrated on c40, which the LLM
  verdict compensates for (section 0: 88% judgment accuracy).
- Open problems: scene localisation (SD_inpaint IoU 0.39, CASIA 0.33) and real images flagged as fake after heavy
  filtering in the harder sets (Test-Challenge 21%, DiffusionFace 13%).

## 5. Self-explaining LLM baselines (`explain/eval_selfexplain.py`)

Every model gets the same open question, *"Is this face real or fake? Explain why."*, with no label and no
parts in the prompt. Test images: DDVQA c40 test (144), and 72 real + 72 fake validation images each from
DiffusionFace (seen generators), CASIA v2 and SD_inpaint, plus 72 fakes from unseen generators (LDM, SD 2.1).
Verdict = first "real"/"fake" in the answer, or the MMTD phrasing ("has (not) been tampered with", "taken directly by the
camera"), which Stage C uses for scene images. For faces, the parts the answer names (skin / cheeks / forehead /
chin, eyes, eyebrows, nose, mouth; "face" alone does not count) are compared with the parts covered by the
ground-truth mask (precision) and by the model's own predicted mask (consistency).

Verdict accuracy:

| Model | DDVQA c40 | DiffusionFace | unseen generators | CASIA v2 | SD_inpaint |
|---|---|---|---|---|---|
| Released M2F2-Det | 0.868 | 0.576 | 0.042 | 0.521 | 0.493 |
| Self-explaining stage 3 from plain LLaVA (section 2 data) | 0.507 | 0.549 | 0.792 | 0.500 | 0.542 |
| Forensic model, step 0 (release LLM + Stage-A v3 heads, function-preserving projector) | **0.889** | 0.597 | 0.056 | 0.597 | 0.521 |
| Forensic model after Stage B (projector only, 150 steps x 32, LLM frozen) | 0.806 | 0.597 | 0.194 | **0.764** | **0.708** |
| **Forensic model after Stage C** (QLoRA r=128, 1 epoch of the 13,378-conversation mix) | 0.785 | **0.757** | **0.611** | 0.722 | 0.590 |

Named parts inside the ground-truth forged region (precision), DDVQA / DiffusionFace: released 0.753 / 0.770,
plain-LLaVA self-explaining 0.603 / 0.793, forensic step 0 0.603 / 0.857.

- The released LLM decides well on its own domain but calls 96% of unseen-generator fakes real and is at chance on
  scene edits, even though (forensic model) the detector inside it scores those fakes at ~0.99: it never learned to
  follow the evidence token outside FF++.
- The self-explaining model trained from plain LLaVA collapsed to one answer ("This image is fake. The eyes appear
  overly small to the face.") for almost every image, with either prompt template. One epoch of LoRA on 12 GB learns
  the wording (train loss 0.33) but not the one-token decision, which falls back to the majority class (66% of DDVQA
  training images are fake). The released model also trained its vision tower and saw ~9x more samples.
- Hence stages B / C start from the released LLM, with the new 16-dim evidence projector initialised from the released
  2-dim projector on the real/fake inputs and zero on the new ones (`scripts/prepare_forensic_base.py`): at step 0
  the model keeps the released behaviour (DDVQA 0.889); training then has to teach it to use the type, filter and
  region evidence.
- Stage B alone (only the evidence projector trains) already moves the frozen LLM towards the detector on scene edits
  (CASIA 0.597 -> 0.764, SD_inpaint 0.521 -> 0.708, unseen generators 0.056 -> 0.194) at some cost on DDVQA
  (0.889 -> 0.806); Stage C (LoRA on the LLM) continues from this projector.
- Stage C (the LLM itself, 1 epoch) writes its own explanations (DDVQA style on faces, MMTD "tampered area / judgment
  basis" style on scenes) and catches far more AI-generated faces than the released model (unseen generators 0.042 ->
  0.611, DiffusionFace 0.576 -> 0.757), but its verdicts still trail the evidence it is given (the v3 heads alone reach
  AUC 0.94-0.99 on these sets, 94% TPR on unseen generators) and it loses 0.08 on DDVQA. In the UI the forensic heads'
  verdict is therefore shown next to the explainer's, with a warning when they disagree. Train loss 0.40, 12.3 h.

## 6. Stage-A v3 and v4

- **v3** (`kaggle/stagea_v3.py`) was meant to use the corrected DiffusionFace index but ran a stale copy of the code
  from the Kaggle dataset (its log shows the old 271,439 DiffusionFace records). It is therefore a second v2 run with
  another seed: per-source differences to v2 are within about +-0.03 (log: `outputs/stagea_v3/log.jsonl`), except
  DDVQA c40 AUC 0.865 (v2 0.845, released detector 0.863).
- **v4** (`kaggle/stagea_v4.py`, code embedded in the job, index checked: 241,439 records, 30,000 Inpaint images with
  masks, no mask files as samples), 14,000 steps. Log: `outputs/stagea_v4/log.jsonl`.

| Test set | v3 (20k steps, old index) | v4 (14k steps, fixed index) |
|---|---|---|
| DiffusionFace inpainting edits: IoU / F1 / pointing | not supervised | 0.227 / 0.337 / 0.565 |
| OpenForensics Test-Dev: IoU / AUC | 0.914 / 0.999 | 0.910 / 0.996 |
| OpenForensics Test-Challenge: IoU / AUC / FPR | 0.753 / 0.930 / 0.176 | 0.714 / 0.920 / 0.256 |
| SD_inpaint: IoU / AUC / FPR | 0.406 / 0.947 / 0.160 | 0.359 / 0.935 / 0.283 |
| CASIA v2: IoU / AUC / FPR | 0.312 / 0.941 / 0.120 | 0.271 / 0.939 / 0.143 |
| DDVQA c40: AUC / FPR | 0.865 / 0.556 | 0.853 / 0.583 |
| DiffusionFace unseen generators: TPR | 0.937 | 0.917 |

v4 learns to localise diffusion face edits, but with fewer steps it is less calibrated (more real images flagged).
Stages B / C use the v3 heads, since the LLM learns to rely on the detector's probabilities; v4 continues as a longer
Stage-A run before it replaces them.

## 7. Stage-A v5: high-res heatmap branch, DINOv2 ablation, type-balanced sampling

`kaggle/stagea_v5.py`, two runs in one Kaggle session (T4 each), 672 px render for a high-res branch (SRM noise
stem, log-FFT feature, 168x168 heatmap), fakes balanced over forgery types. GPU 0 also feeds frozen DINOv2-B
features to the decoder; it ran ~3x slower (2.85 vs 0.95 s/step) and stopped at 8,757 steps (time cap), the
no-DINOv2 run reached 20,000. Logs: `outputs/stagea_v5_dino/`, `outputs/stagea_v5_nodino/`.

v5 without DINOv2 (20k steps) vs v3 (20k steps):

| Test set | v3 | v5 |
|---|---|---|
| DiffusionFace inpainting edits: IoU / pointing | (not supervised) | **0.443 / 0.717** (v4: 0.227 / 0.565) |
| DiffusionFace unseen generators, TPR clean / filtered | 0.937 / 0.790 | **0.970 / 0.977** |
| DiffusionFace AUC filtered | 0.935 | **0.970** |
| FPR on reals: SD_inpaint / DDVQA c40 / OpenForensics Test-Dev | 0.160 / 0.556 / 0.037 | **0.107 / 0.458 / 0.023** |
| CASIA v2 IoU / pointing | 0.312 / 0.590 | **0.345 / 0.630** |
| OpenForensics Test-Dev IoU | **0.914** | 0.899 |
| OpenForensics Test-Challenge IoU / pointing | **0.753 / 0.977** | 0.579 / 0.827 |
| SD_inpaint IoU / pointing | **0.406 / 0.803** | 0.284 / 0.660 |
| DDVQA c40 AUC | **0.865** | 0.787 |

DINOv2 at equal steps (8,000), with / without: SD_inpaint IoU 0.403 / 0.314 (pointing 0.797 / 0.627), CASIA
0.293 / 0.246, DiffusionFace 0.348 / 0.346, OpenForensics Test-Challenge 0.553 / 0.586 (filtered 0.605 / 0.550),
detection AUC about equal. DINOv2 helps localisation of scene edits but costs 3x the time.

The regressions come from the sampling: balancing fakes over types let large sources dominate their type
(OpenForensics fakes ~10% -> ~3%, SD_inpaint ~10% -> ~4%, DDVQA ~10% -> 0.2% of samples). Next (v6): 50/50 mix of
per-source and per-(type, source) balancing (`--balance mix`: every source keeps >= 2.7% of samples, OpenForensics
and DDVQA 6.7%, SD_inpaint 7.5%), with DINOv2 and a full 20k steps.
