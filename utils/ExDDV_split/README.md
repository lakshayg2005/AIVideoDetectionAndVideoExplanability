# ExDDV preprocessing

Turns [ExDDV](https://github.com/vladhondru25/ExDDV) explanations plus the source videos into
frame images mapped to "why this looks fake" text, in the format M2F2-Det's Stage 2/3 training reads.

```bash
bash scripts/exddv_pipeline.sh          # download -> extract and map -> validate
tail -f data/ExDDV/pipeline.log         # watch from another terminal
```

Re-running is safe: the download and finished videos are reused, and failed videos are retried.
Output goes to `data/ExDDV/`, which is git-ignored. No Kaggle account is needed.

## What it does

1. **Picks frames.** Annotators clicked on the artefact while watching, and `click_locations` records
   those frame numbers. We extract those frames rather than sampling uniformly, because they are the
   ones the description refers to. Videos with more than 8 clicks are evenly thinned to 8. The tool
   records each click as the displayed frame + 1, so `frame_idx` is the key minus 1.
2. **Crops faces.** OpenCV YuNet finds the face under the click, or else the largest face. A missed
   detection borrows the box from the closest frame of the same video. Crops are 224x224 with framing
   matched to `utils/DDVQA_images`.
3. **Maps text.** Every image from a video gets that video's description.
4. **Removes train/test overlap.** Train annotations that share a person with test are dropped and
   listed in `reports/removed_train_annotations.csv`. This includes the mirrored pair: `837_890` and
   `890_837` involve the same two people. `--remove-train-overlap-with val test` also clears people
   shared with val, leaving 874 train annotations.
5. **Extracts real frames.** A fake named `890_837` is built on real video `890` and is frame-aligned
   with it, so each real image is the same frame number as its fake. Real crops use the same detector
   and framing. Each real video is in one split: test if any of its fakes is in test, else val, else
   train. Real images get DDVQA's answer, "Based on the learned representation, this image is real."

## Output

| Path | Contents |
| --- | --- |
| `faces/<split>/*.jpg` | Face crops; use as `IMG_FOLDER` |
| `annotations/<split>.json` | LLaVA conversations; use as `DATA_PATH` |
| `frames/<split>/*.jpg` | Uncropped frames, so a different crop doesn't need the videos again |
| `manifest.csv` | One row per image: text, split, source video, frame number, click, face box |
| `reports/` | `summary.json`, `missing_videos.csv`, `failed_videos.csv`, `removed_train_annotations.csv`, `contact_sheet.jpg` |
| `real/` | The same `faces/`, `frames/`, `annotations/` and `manifest.csv` for real images, plus `summary.json` |

`real/manifest.csv` lists each real image's fakes in `paired_fake_images`. Fake and real samples are in
separate JSON files and image folders; training needs them combined into one.

Each JSON sample has this shape, with the explanation copied verbatim from ExDDV:

```json
{"id": "00991_f000051", "image": "00991_Face2Face_890_837_f000051.jpg", "conversations": [
  {"from": "human", "value": "<image>\n<deepfake>\nDescribe the authenticity of the image. Is the image real or fake?"},
  {"from": "gpt", "value": "There are artifacts on the nostrils. The face is not well blended on the head. The mouth suffers deformations and distortions, with an artifact region around the mouth. Based on the learned representation, this image is fake."}]}
```

## Results with the Kaggle videos

| Split | Fake annotations | Fake images | Real videos | Real images |
| --- | --- | --- | --- | --- |
| train | 1,561 | 7,071 | 419 | 5,548 |
| val | 269 | 1,182 | 205 | 955 |
| test | 271 | 1,222 | 243 | 1,216 |
| **total** | **2,101** | **9,475** | **867** | **7,719** |

Fake images by manipulation: FaceShifter 3,728, FaceSwap 3,043, Face2Face 2,704. Output is about 2.7 GB.

No person in test appears in train, as a fake or as a real video. Removing 892 train annotations left
133 real videos with no fakes, so they are not extracted.

## Limitations

- **Only FF++ is covered by the Kaggle dataset.** It provides 2,993 of ExDDV's 4,347 annotations. To add
  the rest, get the videos and pass `--deeperforensics-root`, `--dfdc-root` or `--biodeepav-root`.
  ExDDV's `DeepfakeDetection` rows are DFDC videos, not Kaggle's `DeepFakeDetection` folder.
- **Val still overlaps.** Val scores can be optimistic, so use the test score for final results:
  - 308 train annotations share a video pair with val, and 687 share a person with it.
  - 49 val annotations share a video pair with test. `video_in_multiple_splits` in `manifest.csv` marks
    videos in more than one split.
- **Some fakes have no real pair.** 1,605 of the 9,475 fake images belong to a real video placed in a
  different split, so their real frames aren't extracted.
- **Real images have no descriptions.** ExDDV only describes fakes.

## Files here

- `ExDDV.csv`: from ExDDV commit `8b3498e`. CC BY-NC-SA 4.0; cite Hondru et al., arXiv:2503.14421.
- `face_detection_yunet_2023mar.onnx`: from [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
