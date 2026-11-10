# SPLIT: training-free video detection (branch `split-training-free`)

Paper: *SPLIT: Training-Free AI-Generated and Partially Edited Video Detection via Spatial Patch-Level Incoherence and Temporal
Roughness*, J. Hyun and H. Kim, ECCV 2026, arXiv 2607.02886. Official code: https://github.com/mldljyh/SPLIT (MIT licence).

## What it is
No model is trained. A frozen vision encoder (XCLIP-B/16 by default, also CLIP, DINOv2) turns every frame into patch tokens; two
statistics of the token trajectories are combined and compared with a threshold:

| | meaning | formula (per clip) |
|---|---|---|
| **TTR** two-step temporal roughness | how irregular each patch's token path is (flicker, texture refresh, local misalignment) | mean over patches of log2(L1 + eps) - log2(L2 + eps); L1 = path length of one-step changes, L2 = normalised length of two-step chords |
| **LSMI** local spatial motion incoherence | whether neighbouring patches change *together* (real motion) or not (local edits) | mean norm of the spatial gradients of the motion field M_t = P_{t+1} - P_t |
| **score** | | TTR^8 * LSMI (gamma = 8, fixed) |

A clip is called manipulated if `score >= tau`. `tau` is **calibrated on real videos only**: the smallest threshold at which at most a
fraction alpha (0.1 %, 1 %, 5 %) of real videos is flagged. So the "thresholds" you suspected are the whole decision rule; the
only fitted numbers are those thresholds. Preprocessing: 8 fps, up to 2 s (16 frames), 10 % cropped from both ends of the longer
side, 224 x 224, ImageNet normalisation, channels in BGR order exactly as the official loader reads them (`--rgb` to change).

## Use
```bash
python -m video.split score clip.mp4                       # TTR, LSMI, score of the first 2 s
python -m video.split score clip.mp4 --windows             # every 2 s window (a score timeline)
python -m video.split calibrate --real-dir REAL_VIDEOS --out cal.json
python -m video.split score clip.mp4 --calibration cal.json --alpha 0.01
python -m video.split eval --real-dir REAL --real-dir2 REAL_B --fake-dir FAKE --out report.json
```
`eval` reports Fake Recall at FPR 0.1 / 1 / 5 % with thresholds from each real set and their harmonic mean (the paper's cross-real
threshold transfer) plus AUROC. `video.split.window_scores` returns per-window records with a per-patch TTR grid, which shows
roughly *where* the roughness is.

## Checks
- `python tests/test_split.py`: formulas against the paper's equations written out with loops, preprocessing, calibration, evaluation.
- `python tests/test_split_vs_official.py`: our TTR, LSMI and score against the official `SPLIT_model` on identical input and weights
  (XCLIP-16, T = 8 and 16): worst relative difference 2.8e-6 (the official code uses 0.693147 for ln 2); the official thresholding
  rule is reproduced.

## What is not done
- No benchmark number yet: it needs real and fake videos (FakeParts, GenVideo-Val, ViF-Bench; for our own test the FF++ fakes) and a
  real-only calibration set that is not the test set.
- The paper's thresholds are not transferable to other encoders or other real-video domains: calibrate on real videos of the
  deployment domain.
- It is not wired into the Part 2 timeline / Part 6 UI of the video pipeline; `window_scores` is the hook for that.
- SPLIT detects temporal roughness; a clean face swap that is temporally smooth may score low. That is what the evaluation would tell.
