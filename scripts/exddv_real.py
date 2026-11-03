"""Extract the real frames that correspond to the fake images from exddv_preprocess.py.

An FF++ fake named 890_837 is built on real video 890 (original/890.mp4) and stays frame-aligned
with it, so each real image is the same frame number a fake image was taken from. Frames shared
by several manipulations are saved once. Crops use the same detector and framing as the fakes,
so preprocessing can't become a clue that separates the two classes.

A real video goes in one split only: test if any of its fakes is in test, else val, else train.
It only uses frames from fakes in that split, so a fake in another split has no real pair.

Output under --out/real:
  faces/<split>/real_<video>_f<frame>.jpg   crops   (M2F2 image_folder)
  frames/<split>/<same name>.jpg            full frames
  annotations/<split>.json                  LLaVA conversations, same prompt as the fakes
  manifest.csv                              one row per image, with its paired fake images
  summary.json
"""
import argparse
import json
import multiprocessing as mp
import os

import cv2
import pandas as pd

from exddv_preprocess import PROMPT, SPLITS, detect_face, init_worker, read_frames, square_box

# DDVQA's answer for real images. ExDDV has no descriptions of real videos, so none is invented.
ANSWER = "Based on the learned representation, this image is real."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ffpp-root", required=True, help="folder holding original/")
    p.add_argument("--out", default="data/ExDDV", help="exddv_preprocess.py output folder")
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--face-model", default="utils/ExDDV_split/face_detection_yunet_2023mar.onnx")
    p.add_argument("--min-face-score", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def plan_real_videos(fakes):
    """Return one task per real video: its split and, per frame, the clicks of the fakes it pairs with."""
    fakes = fakes[fakes["dataset"] == "Farceforensics++"].copy()
    fakes["real_video"] = fakes["movie_name"].str.split("_").str[0]
    tasks = []
    for video, group in fakes.groupby("real_video"):
        split = next(s for s in ("test", "val", "train") if (group["split"] == s).any())
        frames = {}
        for r in group[group["split"] == split].itertuples():
            entry = frames.setdefault(int(r.frame_idx), {"click_x": r.click_x, "click_y": r.click_y, "fakes": []})
            entry["fakes"].append(os.path.basename(r.frame_image))
        tasks.append({"video": video, "split": split, "frames": frames, "n_fake_images": len(group),
                      "n_paired_fake_images": sum(len(f["fakes"]) for f in frames.values())})
    return tasks


def process_video(job):
    """Extract one real video's frames. Returns a state dict; never raises, so one bad video can't stop the run."""
    task, args = job
    state_path = os.path.join(args["out"], "real", ".state", f"{task['video']}.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)

    result = {"video": task["video"], "split": task["split"], "records": [], "error": None,
              "frames_past_video_end": 0, "n_fake_images": task["n_fake_images"],
              "n_paired_fake_images": task["n_paired_fake_images"]}
    try:
        path = os.path.join(args["ffpp_root"], "original", f"{task['video']}.mp4")
        frames, _ = read_frames(path, list(task["frames"]))
        result["frames_past_video_end"] = len(task["frames"]) - len(frames)
        boxes = {i: detect_face(frames[i], task["frames"][i]["click_x"], task["frames"][i]["click_y"],
                                args["face_model"], args["min_face_score"]) for i in frames}
        found = [i for i, b in boxes.items() if b is not None]
        params = [cv2.IMWRITE_JPEG_QUALITY, args["jpeg_quality"]]
        for i in sorted(frames):
            frame = frames[i]
            fh, fw = frame.shape[:2]
            name = f"real_{task['video']}_f{i:06d}.jpg"
            frame_rel = os.path.join("frames", task["split"], name)
            cv2.imwrite(os.path.join(args["out"], "real", frame_rel), frame, params)
            rec = {"split": task["split"], "label": "real", "real_video": task["video"], "frame_idx": i,
                   "frame_image": frame_rel, "face_image": "", "face_source": "none", "frame_w": fw, "frame_h": fh,
                   "paired_fake_images": ";".join(task["frames"][i]["fakes"])}
            if found:
                # A missed detection borrows the box from the closest frame of the same video.
                source = i if boxes[i] else min(found, key=lambda f: abs(f - i))
                x0, y0, side = square_box(boxes[source], fw, fh, args["margin"])
                crop = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (args["crop_size"], args["crop_size"]),
                                  interpolation=cv2.INTER_AREA if side > args["crop_size"] else cv2.INTER_CUBIC)
                face_rel = os.path.join("faces", task["split"], name)
                cv2.imwrite(os.path.join(args["out"], "real", face_rel), crop, params)
                rec.update(face_image=face_rel, face_source="detected" if boxes[i] else "nearest_frame",
                           face_x0=x0, face_y0=y0, face_side=side)
            result["records"].append(rec)
    except Exception as exc:
        # Not saved to .state, so the next run retries it.
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    tmp = f"{state_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, state_path)
    return result


def main():
    args = parse_args()
    real_dir = os.path.join(args.out, "real")
    for sub in (".state", "annotations", *[f"{k}/{s}" for k in ("faces", "frames") for s in SPLITS]):
        os.makedirs(os.path.join(real_dir, sub), exist_ok=True)

    tasks = plan_real_videos(pd.read_csv(os.path.join(args.out, "manifest.csv"), keep_default_na=False))
    # A task's plan changes if the fake set changes, so stale results are never reused.
    plan_path = os.path.join(real_dir, ".state", "plan.json")
    plan = json.dumps(tasks, sort_keys=True)
    if os.path.exists(plan_path) and open(plan_path).read() != plan:
        raise SystemExit(f"The fake images changed since {real_dir} was built. Delete that folder and re-run.")
    with open(plan_path, "w") as f:
        f.write(plan)

    worker_args = {**{k: getattr(args, k) for k in ("out", "ffpp_root", "crop_size", "margin", "jpeg_quality",
                                                    "face_model", "min_face_score")}}
    results = []
    with mp.Pool(args.workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_video, [(t, worker_args) for t in tasks], chunksize=4), 1):
            results.append(res)
            if i % 100 == 0 or i == len(tasks):
                print(f"REAL_PROGRESS {i}/{len(tasks)} errors={sum(bool(r['error']) for r in results)}", flush=True)

    manifest = pd.DataFrame([rec for r in results for rec in r["records"]]).sort_values(["split", "real_video", "frame_idx"])
    usable = manifest[manifest["face_image"].fillna("") != ""]
    usable.to_csv(os.path.join(real_dir, "manifest.csv"), index=False)
    for split in SPLITS:
        items = [{"id": f"real_{r.real_video}_f{r.frame_idx:06d}", "image": os.path.basename(r.face_image),
                  "conversations": [{"from": "human", "value": PROMPT}, {"from": "gpt", "value": ANSWER}]}
                 for r in usable[usable["split"] == split].itertuples()]
        with open(os.path.join(real_dir, "annotations", f"{split}.json"), "w") as f:
            json.dump(items, f, indent=1)

    summary = {
        "real_videos": len(results),
        "real_videos_by_split": pd.Series([r["split"] for r in results]).value_counts().to_dict(),
        "failed_videos": {r["video"]: r["error"] for r in results if r["error"]},
        "images": len(usable),
        "images_by_split": usable["split"].value_counts().to_dict(),
        "face_source": manifest["face_source"].value_counts().to_dict(),
        "frames_past_real_video_end": sum(r["frames_past_video_end"] for r in results),
        "fake_images": sum(r["n_fake_images"] for r in results),
        "fake_images_with_a_real_pair": int(usable["paired_fake_images"].str.split(";").str.len().sum()),
    }
    with open(os.path.join(real_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("REAL_DONE", flush=True)


if __name__ == "__main__":
    main()
