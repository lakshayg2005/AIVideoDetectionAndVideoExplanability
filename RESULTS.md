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
