# Our video pipeline vs SPLIT on ViF-Bench (same 152 videos)

76 real + 76 fake (4 clips from each of 19 video generators: CogVideoX, HunyuanVideo, LTX-Video, SkyReels, Wan 2.1/2.2, Gen-4 Turbo, Hailuo, Kling, Pika, PixVerse, Sora 2), balanced. Positive class = fake.
These are fully AI-generated videos, not face swaps; our face-based pipeline was not trained for them (no ViF-Bench data used in training).

- **Our pipeline**: Part 5 of the explainability-grounding branch (v7 forensic heads, 16 sampled frames, Part 2 verdict rules; `uncertain` counts as not fake). AUROC from the segment score.
- **SPLIT**: threshold calibrated on the 89 other real clips at the stated real false-positive rate, first 2 s of each video.

| Detector | Accuracy | Fake recall | Precision | F1 | Real called fake | AUROC | TP/FP/FN/TN |
|---|---|---|---|---|---|---|---|
| **Our pipeline** | 53.3% | 30.3% | 56.1% | 0.39 | 23.7% | 0.60 | 23/18/53/58 |
| SPLIT XCLIP-16, 5% real FPR | 46.1% | 2.6% | 20.0% | 0.05 | 10.5% | 0.38 | 2/8/74/68 |
| SPLIT XCLIP-16, 10% real FPR | 42.1% | 11.8% | 30.0% | 0.17 | 27.6% | 0.38 | 9/21/67/55 |
| SPLIT XCLIP-16, 20% real FPR | 40.1% | 17.1% | 31.7% | 0.22 | 36.8% | 0.38 | 13/28/63/48 |
| SPLIT DINO-base, 5% real FPR | 49.3% | 2.6% | 40.0% | 0.05 | 3.9% | 0.38 | 2/3/74/73 |
| SPLIT DINO-base, 10% real FPR | 45.4% | 10.5% | 34.8% | 0.16 | 19.7% | 0.38 | 8/15/68/61 |
| SPLIT DINO-base, 20% real FPR | 40.8% | 17.1% | 32.5% | 0.22 | 35.5% | 0.38 | 13/27/63/49 |

AUROC 0.5 = chance, below 0.5 = ranks the wrong way round. With 76 videos per class and 4 per generator, differences of a few points are noise, and per-generator numbers are far too small to read (4 videos each).
Our full report: `report.md` / `results.csv` in the `eval_part5` output.
