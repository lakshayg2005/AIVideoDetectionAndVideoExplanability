# Video pipeline plan

The image model (Stage A heads + Stage B/C explainer) is the engine; the video pipeline samples frames, tracks faces,
runs the image model per face crop, and turns the per-frame evidence into a timeline, overlays and one explanation.
Built in parts; each part works and is tested on its own.

| Part | What | Files | Status |
|---|---|---|---|
| 1 | Scene-aware frame sampling, MediaPipe face detection + tracking, per-frame forensic records (JSONL) | `video/frames.py`, `video/faces.py`, `video/analyze.py`, `tests/test_video_part1.py` | done |
| 2 | Temporal aggregation: smoothed fake-probability timeline per face track, suspect segments, video verdict | `video/timeline.py` | |
| 3 | Overlays: heatmap video / frame strip and a timeline plot | `video/render.py` | |
| 4 | Video-level explanation: LLM over the key frames with the timeline summary as the text hint | `video/explain.py` | |
| 5 | Video evaluation on held-out videos (FF++ test; ExDDV stays separate) | `video/eval.py` | |
| 6 | UI: video tab wired to parts 1-5 | `app/forensic_webui.py` | |

Part 1 usage:

    python -m video.analyze clip.mp4 --stagea runs/stagea_v7/last.pt \
        --m2f2-shard checkpoints/llava-v1.5-7b-M2F2-Det/model-00003-of-00003.safetensors --out clip.frames.jsonl

Output: first line is the video header (fps, size, duration, shot cuts), then one line per (sampled frame, face track):
`t, frame, scene, cut, track, box, p_fake, type, types, filters, forgery_area`. Track id -1 means no face was found
and the whole frame was analysed. Check: `python tests/test_video_part1.py` (needs no model).

Design notes:
- Frames are sampled sparsely, so the tracker matches by centre distance (in face sizes), not overlap alone.
- Tracks restart at shot cuts; the first frames of every shot are always sampled, so a short manipulated shot
  is not skipped by an even grid.
- Crops use the same framing as the image UI (square, 35% margin), the framing the heads were trained on.
