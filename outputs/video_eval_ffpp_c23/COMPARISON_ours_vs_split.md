# Our video pipeline vs SPLIT on FF++ c23 (same 120 test videos)

60 real (`original`) + 60 fake (12 each of Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures), balanced, so accuracy is meaningful.
Positive class = fake. FF++ was not used for training.

- **Our pipeline**: Part 5 of the explainability-grounding branch (v7 forensic heads, 16 sampled frames, Part 2 verdict rules). A video counts as fake only if the verdict is `fake`; `uncertain` counts as not fake. AUROC from the segment score.
- **SPLIT** (training-free, split-training-free branch): threshold calibrated on the 80 other real videos at the stated real false-positive rate, first 2 s of each video. AUROC is threshold-free.

| Detector | Accuracy | Fake recall | Precision | F1 | Real called fake | AUROC | TP/FP/FN/TN |
|---|---|---|---|---|---|---|---|
| **Our pipeline** | 55.8% | 43.3% | 57.8% | 0.50 | 31.7% | 0.62 | 26/19/34/41 |
| SPLIT XCLIP-16, 5% real FPR | 48.3% | 5.0% | 37.5% | 0.09 | 8.3% | 0.53 | 3/5/57/55 |
| SPLIT XCLIP-16, 10% real FPR | 46.7% | 8.3% | 35.7% | 0.14 | 15.0% | 0.53 | 5/9/55/51 |
| SPLIT XCLIP-16, 20% real FPR | 52.5% | 26.7% | 55.2% | 0.36 | 21.7% | 0.53 | 16/13/44/47 |
| SPLIT DINO-base, 5% real FPR | 50.0% | 5.0% | 50.0% | 0.09 | 5.0% | 0.54 | 3/3/57/57 |
| SPLIT DINO-base, 10% real FPR | 52.5% | 16.7% | 58.8% | 0.26 | 11.7% | 0.54 | 10/7/50/53 |
| SPLIT DINO-base, 20% real FPR | 50.8% | 28.3% | 51.5% | 0.37 | 26.7% | 0.54 | 17/16/43/44 |

AUROC 0.5 = chance. With 60 videos per class a few points of difference is noise; SPLIT's AUROC of 0.53-0.54 is chance level, ours (0.62) is weak but above it.
Raw numbers: `compare_ours_vs_split.json`. Our full report: `report.md` / `results.csv`.
