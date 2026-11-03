"""Turn ExDDV annotations + source videos into frame images mapped to explanations.

For every ExDDV annotation (one video + one free-text "why it looks fake"):
  1. Pick the frames the annotator actually clicked on while describing the fake.
  2. Save each full frame, plus a square face crop for M2F2-Det.
  3. Map the description to every image, in a flat manifest and in the LLaVA
     conversation JSON that scripts/finetune_stage_*.sh reads.

Rows whose source video is not available are skipped and listed in
reports/missing_videos.csv. Re-running resumes: finished videos are not re-decoded.

Output layout (under --out):
  faces/<split>/<id>_<manipulation>_<video>_f<frame>.jpg   crops   (M2F2 image_folder)
  frames/<split>/<same name>.jpg                           full frames
  annotations/<split>.json                                 LLaVA conversations
  manifest.csv                                             one row per image
  reports/summary.json, missing_videos.csv, failed_videos.csv, removed_train_annotations.csv
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
PROMPT = "<image>\n<deepfake>\nDescribe the authenticity of the image. Is the image real or fake?"
# Same closing sentence as the DDVQA training JSON in utils/DDVQA_split.
VERDICT = "Based on the learned representation, this image is fake."
DETECT_MAX_SIDE = 720

_detector = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="utils/ExDDV_split/ExDDV.csv")
    p.add_argument("--out", default="data/ExDDV")
    p.add_argument("--ffpp-root", required=True, help="folder holding Face2Face/, FaceShifter/, FaceSwap/")
    p.add_argument("--deeperforensics-root", help="folder holding end_to_end/ (not in the Kaggle dataset)")
    p.add_argument("--dfdc-root", help="folder holding dfdc_train_part_*/ (not in the Kaggle dataset)")
    p.add_argument("--biodeepav-root", help="folder holding the BioDeepAV fake videos (not in the Kaggle dataset)")
    p.add_argument("--max-frames-per-video", type=int, default=8, help="evenly thin videos with more clicks")
    p.add_argument("--crop-size", type=int, default=224, help="face crop side; 224 matches utils/DDVQA_images")
    p.add_argument("--margin", type=float, default=0.1, help="context around the face; 0.1 matches utils/DDVQA_images framing")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--face-model", default="utils/ExDDV_split/face_detection_yunet_2023mar.onnx")
    p.add_argument("--min-face-score", type=float, default=0.7)
    p.add_argument("--remove-train-overlap-with", nargs="*", choices=("val", "test"), default=["test"],
                   help="drop train annotations sharing a person with these splits; pass no value to keep them")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, help="process only the first N available videos (smoke test)")
    return p.parse_args()


def resolve_video(row, args):
    """Mirror the folder rules of ExDDV's src/dataset.py. Returns None if the source isn't configured."""
    dataset, manipulation, name = row["dataset"], row["manipulation"], row["movie_name"]
    # "Farceforensics++" is ExDDV's own spelling. "DeepfakeDetection" means DFDC here,
    # NOT the FF++/Kaggle folder of the same name, which holds different videos.
    if dataset == "Farceforensics++":
        return os.path.join(args.ffpp_root, manipulation, name)
    if dataset == "Deeperforensics" and args.deeperforensics_root:
        return os.path.join(args.deeperforensics_root, manipulation, name)
    if dataset == "DeepfakeDetection" and args.dfdc_root:
        return os.path.join(args.dfdc_root, manipulation, name)
    if dataset == "BioDeepAV" and args.biodeepav_root:
        return os.path.join(args.biodeepav_root, name)
    return None


def select_clicks(click_json, max_frames):
    """Return [(frame_idx, x, y)] sorted by frame, evenly thinned to max_frames.

    ExDDV's annotation tool increments its counter right after showing a frame, so a
    click's key is the displayed frame + 1. frame_idx is the 0-based frame actually shown.
    """
    clicks = sorted((int(k) - 1, float(v["x"]), float(v["y"])) for k, v in json.loads(click_json).items())
    if len(clicks) > max_frames:
        keep = np.linspace(0, len(clicks) - 1, max_frames).round().astype(int)
        clicks = [clicks[i] for i in keep]
    return clicks


def read_frames(path, wanted):
    """Decode sequentially so frame numbers match the annotation tool, which also counted reads."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video")
    frames, idx, last = {}, 0, max(wanted)
    wanted = set(wanted)
    while idx <= last:
        if not cap.grab():
            break
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames[idx] = frame
        idx += 1
    cap.release()
    return frames, idx


def detect_face(frame, click_x, click_y, model_path, min_score):
    """Return an (x, y, w, h) face box: the one under the click if any, else the largest."""
    global _detector
    h, w = frame.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    size = (small.shape[1], small.shape[0])
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(model_path, "", size, score_threshold=min_score)
    _detector.setInputSize(size)
    _, faces = _detector.detect(small)
    if faces is None or len(faces) == 0:
        return None
    px, py = click_x * size[0], click_y * size[1]
    boxes = [tuple(float(v) for v in f[:4]) for f in faces]
    under_click = [b for b in boxes if b[0] <= px <= b[0] + b[2] and b[1] <= py <= b[1] + b[3]]
    x, y, bw, bh = max(under_click or boxes, key=lambda b: b[2] * b[3])
    return x / scale, y / scale, bw / scale, bh / scale


def square_box(box, frame_w, frame_h, margin):
    """Grow a face box by margin and shift it inside the frame, keeping it square so crops aren't distorted."""
    x, y, w, h = box
    side = int(round(min(max(w, h) * (1 + 2 * margin), frame_w, frame_h)))
    x0 = int(np.clip(round(x + w / 2 - side / 2), 0, frame_w - side))
    y0 = int(np.clip(round(y + h / 2 - side / 2), 0, frame_h - side))
    return x0, y0, side


def process_video(task):
    """Extract one annotation's images. Returns a state dict; never raises, so one bad video can't stop the run."""
    row, args = task
    state_path = os.path.join(args["out"], ".state", f"{row['id']}.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)

    result = {"id": row["id"], "records": [], "error": None, "dropped_clicks": 0}
    try:
        clicks = select_clicks(row["click_locations"], args["max_frames_per_video"])
        frames, n_read = read_frames(row["video_path"], [c[0] for c in clicks])
        result["dropped_clicks"] = len(clicks) - len(frames)
        clicks = [c for c in clicks if c[0] in frames]
        if not clicks:
            raise ValueError(f"no clicked frame inside video ({n_read} frames decoded)")

        boxes = {c[0]: detect_face(frames[c[0]], c[1], c[2], args["face_model"], args["min_face_score"]) for c in clicks}
        found = [f for f, b in boxes.items() if b is not None]
        stem = os.path.splitext(row["movie_name"])[0]
        name_base = f"{row['id']:05d}_{row['manipulation']}_{stem}"
        params = [cv2.IMWRITE_JPEG_QUALITY, args["jpeg_quality"]]

        for frame_idx, cx, cy in clicks:
            frame = frames[frame_idx]
            fh, fw = frame.shape[:2]
            name = f"{name_base}_f{frame_idx:06d}.jpg"
            frame_rel = os.path.join("frames", row["split"], name)
            cv2.imwrite(os.path.join(args["out"], frame_rel), frame, params)

            rec = {"frame_idx": frame_idx, "frame_image": frame_rel, "frame_w": fw, "frame_h": fh,
                   "click_x": cx, "click_y": cy, "face_image": "", "face_source": "none",
                   "face_x0": "", "face_y0": "", "face_side": "", "click_x_crop": "", "click_y_crop": ""}
            if found:
                # A missed detection borrows the box from the closest frame of the same video.
                source_frame = frame_idx if boxes[frame_idx] else min(found, key=lambda f: abs(f - frame_idx))
                x0, y0, side = square_box(boxes[source_frame], fw, fh, args["margin"])
                crop = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (args["crop_size"], args["crop_size"]),
                                  interpolation=cv2.INTER_AREA if side > args["crop_size"] else cv2.INTER_CUBIC)
                face_rel = os.path.join("faces", row["split"], name)
                cv2.imwrite(os.path.join(args["out"], face_rel), crop, params)
                crop_x, crop_y = (cx * fw - x0) / side, (cy * fh - y0) / side
                click_in_crop = 0 <= crop_x <= 1 and 0 <= crop_y <= 1
                rec.update(face_image=face_rel, face_source="detected" if boxes[frame_idx] else "nearest_frame",
                           face_x0=x0, face_y0=y0, face_side=side,
                           # Blank when the click falls outside the crop (e.g. on the neck).
                           click_x_crop=round(crop_x, 5) if click_in_crop else "",
                           click_y_crop=round(crop_y, 5) if click_in_crop else "")
            result["records"].append(rec)
    except Exception as exc:
        # Not saved to .state, so the next run retries it (e.g. after re-downloading the video).
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    tmp = f"{state_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, state_path)  # atomic: a killed run never leaves a half-written state file
    return result


def people_in_video(row):
    """FF++ names are <target>_<source> video ids. Other datasets have no ids, so the video stands in."""
    stem = os.path.splitext(row["movie_name"])[0]
    return set(stem.split("_")) if row["dataset"] == "Farceforensics++" else {stem}


def drop_train_overlap(df, holdout_splits):
    """Split df into (kept, removed), removing train rows that share a person with a held-out split.

    ExDDV's official split can put one FF++ pair in several splits under different manipulations,
    and the mirrored pair (837_890 vs 890_837) involves the same two people.
    """
    people = df.apply(people_in_video, axis=1)
    held_out = set().union(*people[df["split"].isin(holdout_splits)])
    overlap = (df["split"] == "train") & people.map(lambda p: bool(p & held_out))
    return df[~overlap].copy(), df[overlap].copy()


def discard_outputs(out, ids):
    """Delete images and resume state of annotations an earlier run extracted but this one excludes."""
    prefixes = tuple(f"{i:05d}_" for i in ids)
    if not prefixes:
        return
    for kind in ("faces", "frames"):
        folder = os.path.join(out, kind, "train")
        for name in os.listdir(folder):
            if name.startswith(prefixes):
                os.remove(os.path.join(folder, name))
    for i in ids:
        state = os.path.join(out, ".state", f"{i}.json")
        if os.path.exists(state):
            os.remove(state)


def init_worker():
    cv2.setNumThreads(1)  # parallelism comes from the worker processes


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def ensure_sentence(text):
    return text if text[-1] in ".!?" else text + "."


def write_outputs(df, results, args, n_removed):
    """Build manifest, LLaVA JSONs and reports from the per-video state."""
    meta = df.set_index("id")
    # Flagged rather than fixed: ExDDV's official split puts the same FF++ pair
    # (e.g. 890_837) in several splits under different manipulations.
    multi_split = df.groupby("movie_name")["split"].transform("nunique") > 1
    leaky_ids = set(df.loc[multi_split, "id"])

    rows, failed = [], []
    for res in results:
        row = meta.loc[res["id"]]
        if res["error"]:
            failed.append({"id": res["id"], "video_path": row["video_path"], "error": res["error"]})
        for rec in res["records"]:
            rows.append({"exddv_id": res["id"], "split": row["split"], "label": "fake", "dataset": row["dataset"],
                         "manipulation": row["manipulation"], "movie_name": row["movie_name"],
                         "video_in_multiple_splits": res["id"] in leaky_ids,
                         "annotator": row["username"], "difficulty": row["difficulty"],
                         "explanation": clean_text(row["text"]), **rec})
    if not rows:
        sys.exit(f"All {len(failed)} videos failed, first error: {failed[0]['error'] if failed else 'none'}")
    manifest = pd.DataFrame(rows).sort_values(["split", "exddv_id", "frame_idx"])
    manifest.to_csv(os.path.join(args.out, "manifest.csv"), index=False)

    for split in SPLITS:
        items = []
        for r in manifest[(manifest["split"] == split) & (manifest["face_image"] != "")].itertuples():
            answer = f"{ensure_sentence(r.explanation)} {VERDICT}"
            items.append({
                "id": f"{r.exddv_id:05d}_f{r.frame_idx:06d}",
                # Relative to faces/<split>, the value scripts/finetune_stage_*.sh passes as IMG_FOLDER.
                "image": os.path.basename(r.face_image),
                "conversations": [{"from": "human", "value": PROMPT}, {"from": "gpt", "value": answer}],
            })
        with open(os.path.join(args.out, "annotations", f"{split}.json"), "w") as f:
            json.dump(items, f, indent=1)

    reports = os.path.join(args.out, "reports")
    missing = df[df["video_path"].isna() | ~df["video_path"].fillna("").map(os.path.exists)]
    missing[["id", "dataset", "manipulation", "movie_name", "split", "video_path"]].to_csv(
        os.path.join(reports, "missing_videos.csv"), index=False)
    pd.DataFrame(failed, columns=["id", "video_path", "error"]).to_csv(
        os.path.join(reports, "failed_videos.csv"), index=False)

    usable = manifest[manifest["face_image"] != ""]
    summary = {
        "annotations_in_csv": len(df) + n_removed,
        "train_annotations_removed_for_overlap": n_removed,
        "annotations_missing_video": len(missing),
        "annotations_missing_video_by_dataset": missing["dataset"].value_counts().to_dict(),
        "annotations_failed": len(failed),
        "annotations_with_images": int(manifest["exddv_id"].nunique()),
        "clicks_outside_video_dropped": sum(r["dropped_clicks"] for r in results),
        "full_frames": len(manifest),
        "face_crops": len(usable),
        "face_source": manifest["face_source"].value_counts().to_dict(),
        "face_crops_by_split": usable["split"].value_counts().to_dict(),
        "face_crops_by_manipulation": usable["manipulation"].value_counts().to_dict(),
        "annotations_by_split": manifest.drop_duplicates("exddv_id")["split"].value_counts().to_dict(),
        "annotations_whose_video_is_in_multiple_splits": int(
            manifest.drop_duplicates("exddv_id")["video_in_multiple_splits"].sum()),
    }
    with open(os.path.join(reports, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df["video_path"] = df.apply(lambda r: resolve_video(r, args), axis=1)

    for sub in ("annotations", "reports", ".state", *[f"{k}/{s}" for k in ("faces", "frames") for s in SPLITS]):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    df, removed = drop_train_overlap(df, args.remove_train_overlap_with)
    discard_outputs(args.out, removed["id"].tolist())
    removed[["id", "dataset", "manipulation", "movie_name"]].to_csv(
        os.path.join(args.out, "reports", "removed_train_annotations.csv"), index=False)

    available = df[df["video_path"].notna() & df["video_path"].fillna("").map(os.path.exists)]
    if available.empty:
        sys.exit(f"No videos found under --ffpp-root {args.ffpp_root!r}. Check the path.")
    if args.limit:
        available = available.head(args.limit)
    print(f"annotations={len(df)} available={len(available)} "
          f"by_dataset={Counter(available['dataset'])}", flush=True)

    worker_args = {k: getattr(args, k) for k in ("out", "max_frames_per_video", "crop_size", "margin", "jpeg_quality",
                                                "face_model", "min_face_score")}
    tasks = [(r, worker_args) for r in available.to_dict("records")]
    results, errors, start = [], 0, time.time()
    with mp.Pool(args.workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_video, tasks, chunksize=4), 1):
            results.append(res)
            errors += bool(res["error"])
            if i % 100 == 0 or i == len(tasks):
                rate = i / (time.time() - start)
                print(f"PROGRESS {i}/{len(tasks)} errors={errors} {rate:.1f} videos/s "
                      f"eta={(len(tasks) - i) / rate / 60:.1f}min", flush=True)

    summary = write_outputs(df, results, args, len(removed))
    print(json.dumps(summary, indent=2), flush=True)
    print("PREPROCESS_DONE", flush=True)


if __name__ == "__main__":
    main()
