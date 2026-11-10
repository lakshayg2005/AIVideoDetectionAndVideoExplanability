# Video pipeline plan

The image model (Stage A heads + Stage B/C explainer) is the engine; the video pipeline samples frames, tracks faces,
runs the image model per face crop, and turns the per-frame evidence into a timeline, overlays and one explanation.
Built in parts; each part works and is tested on its own.

| Part | What | Files | Status |
|---|---|---|---|
| 1 | Scene-aware frame sampling, MediaPipe face detection + tracking, per-frame forensic records (JSONL) | `video/frames.py`, `video/faces.py`, `video/analyze.py`, `tests/test_video_part1.py` | done |
| 2 | Temporal aggregation: smoothed fake-probability timeline per face track, suspect segments, video verdict | `video/timeline.py`, `tests/test_video_part2.py` | done |
| 3 | Overlays: heatmap video / frame strip and a timeline plot | `video/render.py`, `tests/test_video_part3.py` | done |
| 4 | Video-level explanation: LLM over the key frames with the timeline summary as the text hint | `video/explain.py` | |
| 5 | Video evaluation on held-out videos (FF++ test; ExDDV stays separate) | `video/eval.py` | |
| 6 | UI: video tab wired to parts 1-5 | `app/forensic_webui.py` | |

Part 1 usage:

    python -m video.analyze clip.mp4 --stagea runs/stagea_v7/last.pt \
        --m2f2-shard checkpoints/llava-v1.5-7b-M2F2-Det/model-00003-of-00003.safetensors --out clip.frames.jsonl

Output: first line is the video header (fps, size, duration, shot cuts), then one line per (sampled frame, face track):
`t, frame, scene, cut, track, box, p_fake, type, types, filters, forgery_area`. Track id -1 means no face was found
and the whole frame was analysed. Check: `python tests/test_video_part1.py` (needs no model).

Part 2 usage:

    python -m video.timeline clip.frames.jsonl --out clip.timeline.json

Output (JSON): `verdict` (label real / fake / uncertain, trimmed-mean `p_video`, `fake_fraction`, `dominant_type`, `reasons`),
`segments` (start, end, duration, mean / peak p, type, forgery area, frames, track), `isolated_spikes`, per-track `timeline`
(raw and smoothed p) and `mean_filter_probs`. Check: `python tests/test_video_part2.py` (no model needed).

Part 3 usage:

    python -m video.render clip.mp4 clip.frames.jsonl clip.timeline.json --out-dir out/ \
        [--stagea runs/stagea_v7/last.pt --m2f2-shard <model-00003-of-00003.safetensors> --device cpu]

Writes `<stem>.timeline.png` (fake probability per face over time, suspect segments shaded, shot cuts dotted),
`<stem>.annotated.mp4` (every frame with the face box coloured green -> amber -> red by the interpolated score, plus a verdict
bar with the suspect segments and a playhead) and `<stem>.keyframes.png` (the peak frame of every segment and the cleanest
frames, each next to the forgery heatmap of its face crop; the heatmaps need `--stagea` / `--m2f2-shard`, otherwise the crops
are shown plain). Check: `python tests/test_video_part3.py` (needs only the face image used by the Part 1 test).

Part 3 design notes:
- The annotated video is drawn from the sampled records, not from a new model pass: scores and boxes are interpolated between
  samples, and nothing is drawn where the samples are further apart than 1.5 s (the bar stays dark there: no evidence).
- Only the key frames (<= 6) run the model again, so the heatmaps are cheap even on CPU.

Part 2 design notes:
- Scores are smoothed per face track with a Gaussian whose width follows the sample spacing, so sparse sampling is not
  smoothed less than dense sampling.
- A segment is a run of >= 2 consecutive suspect samples (no gap above 1.5 s). One isolated high score is *not* a segment (the
  image heads flag some real faces) but is listed under `isolated_spikes`.
- Verdict: fake if the trimmed mean (the paper's protocol) is >= 0.5, **or** a localized segment is convincing (mean p >= 0.8 over
  >= 1 s or >= 3 samples), so a short manipulated shot is not averaged away by a long untouched one; uncertain if the mean is near
  the threshold or only weak segments exist; otherwise real.
- Known limit: a manipulation shorter than the sampling step can still be missed (use `--max-frames` to sample more densely).

Design notes (Part 1):
- Frames are sampled sparsely, so the tracker matches by centre distance (in face sizes), not overlap alone.
- Tracks restart at shot cuts; the first frames of every shot are always sampled, so a short manipulated shot
  is not skipped by an even grid.
- Crops use the same framing as the image UI (square, 35% margin), the framing the heads were trained on.
