# Video pipeline plan

The image model (Stage A heads + Stage B/C explainer) is the engine; the video pipeline samples frames, tracks faces,
runs the image model per face crop, and turns the per-frame evidence into a timeline, overlays and one explanation.
Built in parts; each part works and is tested on its own.

| Part | What | Files | Status |
|---|---|---|---|
| 1 | Scene-aware frame sampling, MediaPipe face detection + tracking, per-frame forensic records (JSONL) | `video/frames.py`, `video/faces.py`, `video/analyze.py`, `tests/test_video_part1.py` | done |
| 2 | Temporal aggregation: smoothed fake-probability timeline per face track, suspect segments, video verdict | `video/timeline.py`, `tests/test_video_part2.py` | done |
| 3 | Overlays: heatmap video / frame strip and a timeline plot | `video/render.py`, `tests/test_video_part3.py` | done |
| 4 | Video-level explanation: LLM over the key frames with the timeline summary as the text hint | `video/explain.py`, `tests/test_video_part4.py` | done |
| 5 | Video evaluation on held-out videos (FF++ test; ExDDV stays separate) | `video/eval.py`, `tests/test_video_part5.py` | done |
| 6 | UI: video tab wired to parts 1-4 | `app/forensic_webui.py`, `tests/test_video_part6.py` | done |

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
bar with the suspect segments and a playhead) and `<stem>.keyframes.png` (the peak frame of every segment, and only those when the video has segments; the highest-scoring frames for a fake verdict without a segment; the cleanest frames only for a video judged real), each next to the forgery heatmap of its face crop; the heatmaps need `--stagea` / `--m2f2-shard`, otherwise the crops
are shown plain). Check: `python tests/test_video_part3.py` (needs only the face image used by the Part 1 test).

Part 6 usage: `./run_forensic_ui.sh --stage-c7` (or `python -m app.forensic_webui ...`), open the **Video** tab, upload a clip. One click runs
Parts 1-4: face tracking and per-frame heads, the timeline and verdict, then the plot, the annotated video, the key-frame heatmaps, a table of
suspect segments and the explanation. `--no-llm` skips the key-frame explanations (the rule-based text stays). Check:
`python tests/test_video_part6.py` (stub engine, no model).

Part 6 design notes:
- The tab replaced the first prototype (a uniform frame loop with one verdict from the trimmed mean); it now uses the same code as the command-line parts.
- The language model already loaded for the Image tab is reused for the key frames (no second copy in GPU memory); only a Stage-C explainer
  (`--self-explain`) is used for them, because the two-step prompt needs the heads inside it.
- The annotated video is re-encoded to H.264 with ffmpeg so browsers can play it (OpenCV writes mp4v); without ffmpeg the mp4v file is returned.
- Speed: face analysis is the cost (about 1-3 s per sampled frame on the GPU). More frames find shorter edits but take proportionally longer.

Part 5 usage:

    python -m video.eval --root /path/to/ffpp_test --stagea runs/stagea_v7/last.pt --m2f2-shard <shard 3> --out-dir eval_video/
    python -m video.eval --manifest videos.csv --cache-dir eval_video/frames --out-dir eval_video/     # reuse cached Part 1 output

`--root` is a folder tree whose first-level folder names label the videos (`original/`, `real/`, `youtube/` ... are real, any other folder
such as `Deepfakes/` is a fake and its name is the method); `--manifest` is a CSV `video,label[,method,fake_start,fake_end]`. Per video the Part 1
records are cached, so a re-run (or a changed Part 2 rule) does not need the model. Output: `results.csv` (one row per video), `report.json`, `report.md`.

Part 5 design notes:
- Three video scores are compared so the temporal part is measured, not assumed: `mean` (trimmed mean of the frame probabilities, the image-model
  baseline), `max` (strongest smoothed frame score) and `segment` (strongest suspect segment, else the trimmed mean). Report: AUC for each, and for the
  Part 2 verdicts accuracy, fake recall, false positive rate and the share of `uncertain` (counted as neither fake nor real, so it lowers accuracy).
- Per-method recall and AUC; localisation IoU only when the manifest gives the manipulated interval.
- Held-out data only: nothing is trained or tuned here, and an unreadable video is reported as skipped (and counted in `skipped`), never dropped silently.
- Check: `python tests/test_video_part5.py` (cached synthetic records; no model, no video files).

Part 4 usage:

    python -m video.explain clip.mp4 clip.frames.jsonl clip.timeline.json --out clip.explanation.json --no-llm      # rule-based text only
    python -m video.explain clip.mp4 clip.frames.jsonl clip.timeline.json --out clip.explanation.json \
        --model-path /data3/Aditya/bp_data/ckpt/llava-v1.5-7b-M2F2-forensic_stage-c7-lora --model-base /data3/Aditya/bp_data/ckpt/forensic-base-v7 --fp16

Output: JSON (`summary`, `segments` with the LLM's key-frame explanation and whether it agrees, `clean_frame`, `notes`, `disagreements`,
`text`) and the same text next to it as `.txt`. Check: `python tests/test_video_part4.py` (a stub replaces the LLM; no model needed).

Part 4 design notes:
- The verdict, times and edit types come from the Part 2 timeline, so the text cannot contradict the numbers; the LLM only adds the
  image-level explanation of each segment's peak frame (the strongest 3 segments), with the segment's evidence written before its question
  and the same heatmap hint the image UI gives it. A video with no suspect segment gets the cleanest frame explained instead.
- If the LLM's verdict on a key frame disagrees with the timeline, the disagreement is stated in the text and the timeline verdict stands.
- The real LLM path (`make_llm_ask`) is the image UI's two-step prompt; it was not run in the automated build (CPU-only rule), only the stubbed tests.

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
