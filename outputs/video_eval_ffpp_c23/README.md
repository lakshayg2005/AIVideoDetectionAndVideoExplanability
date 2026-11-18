# Our video pipeline on FF++ c23 (held-out test videos)
Part 5 (`python -m video.eval`) on 120 FF++ c23 test videos: 60 real (`original`) + 12 per manipulation method (Deepfakes, Face2Face,
FaceShifter, FaceSwap, NeuralTextures), 16 sampled frames per video, v7 forensic heads (FF++ not used in training).
- `report.md` / `report.json`: AUC of the three video scores, verdict rates, per-method recall
- `results.csv`: one row per video; `frames/`: cached Part 1 records; `part5_manifest.csv`: the exact videos
- `compare_ours_vs_split.json`: this pipeline and SPLIT on the same videos (accuracy, recall, precision, F1, AUROC)
