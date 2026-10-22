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
