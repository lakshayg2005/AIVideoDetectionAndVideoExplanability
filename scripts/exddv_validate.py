"""Check the output of exddv_preprocess.py against the original ExDDV CSV.

Exits non-zero if any check fails. Also renders a contact sheet with the
annotator's clicks drawn on, for checking by eye that frames and text line up.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")

p = argparse.ArgumentParser()
p.add_argument("--csv", default="utils/ExDDV_split/ExDDV.csv")
p.add_argument("--out", default="data/ExDDV")
p.add_argument("--crop-size", type=int, default=224)
args = p.parse_args()

src = pd.read_csv(args.csv).set_index("id")
man = pd.read_csv(os.path.join(args.out, "manifest.csv"), keep_default_na=False, dtype={"movie_name": str})
problems = []


def check(ok, msg):
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        problems.append(msg)


def norm(text):
    return " ".join(str(text).split())


# Mapping: every image carries its own video's text, split and a frame that was really clicked.
s = src.loc[man["exddv_id"]]
check((man["explanation"].values == s["text"].map(norm).values).all(), "explanation matches the CSV row")
check((man["split"].values == s["split"].values).all(), "split matches the CSV row")
check((man["movie_name"].values == s["movie_name"].values).all(), "video name matches the CSV row")
check(all(str(f + 1) in json.loads(c) for f, c in zip(man["frame_idx"], s["click_locations"])),
      "frame_idx is a clicked frame")

# No sample may be written twice.
check(not man["frame_image"].duplicated().any(), "image names are unique")

# Files on disk.
faces = man[man["face_image"] != ""]
check(all(os.path.exists(os.path.join(args.out, f)) for f in man["frame_image"]), "all full frames exist")
bad = [f for f in faces["face_image"]
       if (im := cv2.imread(os.path.join(args.out, f))) is None or im.shape[:2] != (args.crop_size, args.crop_size)]
check(not bad, f"all {len(faces)} face crops decode at {args.crop_size}x{args.crop_size} ({len(bad)} bad)")
on_disk = sum(len(files) for split in SPLITS for _, _, files in os.walk(os.path.join(args.out, "faces", split)))
check(on_disk == len(faces), f"no orphan crops on disk ({on_disk} files)")

# LLaVA JSON: same format as utils/DDVQA_split, resolvable the way train_deepfake.py resolves it.
total = 0
for split in SPLITS:
    with open(os.path.join(args.out, "annotations", f"{split}.json")) as f:
        items = json.load(f)
    total += len(items)
    by_name = faces[faces["split"] == split].set_index(faces[faces["split"] == split]["face_image"].map(os.path.basename))
    check(len(items) == len(by_name), f"{split}.json has one entry per crop ({len(items)})")
    check(all(os.path.exists(os.path.join(args.out, "faces", split, i["image"])) for i in items),
          f"{split}.json images resolve under faces/{split}")
    check(all(i["conversations"][1]["value"].startswith(by_name.loc[i["image"], "explanation"].rstrip(".!?"))
              for i in items), f"{split}.json answers carry that image's explanation")
    check(all(i["conversations"][0]["value"].startswith("<image>\n<deepfake>\n") for i in items),
          f"{split}.json prompts have the image tokens")

print(f"\n{total} training samples across splits; "
      f"{man['exddv_id'].nunique()} annotations; {len(man) - len(faces)} frames without a face crop")

# Contact sheet: fixed seed; red dot = annotator's click, drawn on the crop.
clicked = faces[faces["click_x_crop"] != ""]
sample = clicked.sample(min(24, len(clicked)), random_state=0)
tiles = []
for r in sample.itertuples():
    im = cv2.imread(os.path.join(args.out, r.face_image))
    cv2.circle(im, (int(float(r.click_x_crop) * im.shape[1]), int(float(r.click_y_crop) * im.shape[0])), 5, (0, 0, 255), 2)
    bar = np.full((44, im.shape[1], 3), 255, np.uint8)
    for line, chunk in enumerate((r.explanation[:34], r.explanation[34:68])):
        cv2.putText(bar, chunk, (2, 16 + line * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    tiles.append(np.vstack([im, bar]))
if tiles:
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    path = os.path.join(args.out, "reports", "contact_sheet.jpg")
    cv2.imwrite(path, sheet)
    print("contact sheet:", path)

# Train overlap removal: no person in test may appear in train, as a fake or as a real video.
def people(split):
    ff = man[(man["dataset"] == "Farceforensics++") & (man["split"] == split)]
    return {p for name in ff["movie_name"] for p in os.path.splitext(name)[0].split("_")}


test_people = people("test")
check(not people("train") & test_people, f"no fake train video shares a person with test ({len(test_people)} people)")

# Real images: right source video, right frames, one split per real video, no leftover files.
real = pd.read_csv(os.path.join(args.out, "real", "manifest.csv"), keep_default_na=False, dtype={"real_video": str})
check(not set(real.loc[real["split"] == "train", "real_video"]) & test_people, "no real train video shows a person in test")
check((real["label"] == "real").all(), f"{len(real)} real images all labelled real")
check((real.groupby("real_video")["split"].nunique() == 1).all(), "each real video is in exactly one split")
check(not real["face_image"].duplicated().any(), "real image names are unique")
fake_by_name = man.set_index(man["frame_image"].map(os.path.basename))
pairs_ok = True
for r in real.itertuples():
    for name in r.paired_fake_images.split(";"):
        f = fake_by_name.loc[name]
        if f["movie_name"][:3] != r.real_video or f["frame_idx"] != r.frame_idx or f["split"] != r.split:
            pairs_ok = False
check(pairs_ok, "each real image pairs with fakes of the same real video, frame and split")
bad = [f for f in real["face_image"]
       if (im := cv2.imread(os.path.join(args.out, "real", f))) is None or im.shape[:2] != (args.crop_size, args.crop_size)]
check(not bad, f"all real crops decode at {args.crop_size}x{args.crop_size} ({len(bad)} bad)")
on_disk = sum(len(files) for split in SPLITS for _, _, files in os.walk(os.path.join(args.out, "real", "faces", split)))
check(on_disk == len(real), f"no orphan real crops on disk ({on_disk} files)")
for split in SPLITS:
    with open(os.path.join(args.out, "real", "annotations", f"{split}.json")) as f:
        items = json.load(f)
    expected = set(real.loc[real["split"] == split, "face_image"].map(os.path.basename))
    check({i["image"] for i in items} == expected and len(items) == len(expected), f"real {split}.json matches its {len(items)} crops")
    check(all(i["conversations"][1]["value"].endswith("this image is real.") for i in items), f"real {split}.json answers say real")

if problems:
    sys.exit(f"\n{len(problems)} check(s) failed")
print("\nALL CHECKS PASSED")
