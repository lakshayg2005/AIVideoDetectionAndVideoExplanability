# SPLIT benchmark results (real-only calibration)
- `ffpp_c23/`: FF++ c23 official test split (70 real calibration / 70 real check / 700 fakes): `split_ffpp_<encoder>.json`;
  `compare_ours_vs_split.json` + `ours_part5_*`: our pipeline (explainability-grounding branch) vs SPLIT on the same 120 videos.
- `vif_bench/`: ViF-Bench (165 real / 2,995 generated fakes): `split_full_<encoder>.json`, `split_report.json` (first 285-fake pilot).
Script: `scripts/split_benchmark.py`. Summary and discussion: `docs/SPLIT.md`.
