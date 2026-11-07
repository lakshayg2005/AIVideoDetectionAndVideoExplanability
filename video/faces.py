"""Face detection and tracking over sampled frames.

Faces are detected per frame with MediaPipe (the Haar detector also fires on animals and objects, see
app/forensic_engine.smart_crop), linked into tracks by box overlap (IoU), and the box of each track is smoothed so the
crops fed to the image model stay steady. Crop geometry matches app.engine.crop_face (square, 35% margin), the framing
the forensic heads were trained on. Tracks restart at shot cuts.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

_DET = None


def _detector():
    global _DET
    if _DET is None:
        import mediapipe as mp
        _DET = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return _DET


def detect_faces(rgb):
    """MediaPipe face boxes for one RGB frame: [(x0, y0, x1, y1, score)] in pixels, largest first."""
    H, W = rgb.shape[:2]
    res = _detector().process(rgb)
    out = []
    for d in res.detections or []:
        b = d.location_data.relative_bounding_box
        x0, y0 = max(0.0, b.xmin * W), max(0.0, b.ymin * H)
        x1, y1 = min(float(W), (b.xmin + b.width) * W), min(float(H), (b.ymin + b.height) * H)
        if x1 - x0 >= 24 and y1 - y0 >= 24:
            out.append((x0, y0, x1, y1, float(d.score[0])))
    return sorted(out, key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_box(box, size, margin=0.35):
    """Square crop around a face box with a margin, clipped to the image: (x0, y0, x1, y1) ints."""
    x0, y0, x1, y1 = box[:4]
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    W, H = size
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(W, cx + side / 2)), int(min(H, cy + side / 2)))


@dataclass
class Track:
    id: int
    box: tuple          # smoothed (x0, y0, x1, y1)
    last_t: float


def center_dist(a, b):
    """Distance between box centres in units of the larger box side (0 = same centre)."""
    ax, ay, bx, by = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    return float(np.hypot(ax - bx, ay - by) / side)


class FaceTracker:
    """Frames are sampled sparsely (often 0.5 s or more apart), so a face can move several face-widths between two
    samples and overlap alone fails: a detection joins a track when the boxes overlap (IoU >= iou_thr) or their
    centres are within `max_dist` face sizes, nearest first. Box smoothing only applies across close frames."""

    def __init__(self, iou_thr=0.3, max_dist=1.5, max_gap_s=2.0, smooth=0.6, smooth_gap_s=0.3):
        self.iou_thr, self.max_dist, self.max_gap_s, self.smooth, self.smooth_gap_s = iou_thr, max_dist, max_gap_s, smooth, smooth_gap_s
        self.tracks, self.next_id = [], 0

    def reset(self):
        self.tracks = []

    def update(self, t, boxes):
        """Assign detections to tracks. Returns [(track_id, smoothed_box, score)] in the order of `boxes`."""
        self.tracks = [k for k in self.tracks if t - k.last_t <= self.max_gap_s]
        pairs = sorted((center_dist(k.box, b), ki, bi) for ki, k in enumerate(self.tracks) for bi, b in enumerate(boxes)
                       if iou(k.box, b) >= self.iou_thr or center_dist(k.box, b) <= self.max_dist)
        used_k, used_b, assigned = set(), set(), {}
        for _, ki, bi in pairs:
            if ki in used_k or bi in used_b:
                continue
            used_k.add(ki); used_b.add(bi); assigned[bi] = self.tracks[ki]
        out = []
        for bi, b in enumerate(boxes):
            if bi in assigned:
                k = assigned[bi]
                a = self.smooth if t - k.last_t <= self.smooth_gap_s else 0.0
                k.box = tuple(a * o + (1 - a) * n for o, n in zip(k.box, b[:4]))
                k.last_t = t
            else:
                k = Track(self.next_id, tuple(b[:4]), t)
                self.next_id += 1
                self.tracks.append(k)
            out.append((k.id, k.box, b[4]))
        return out


def face_crops(frame, tracker, whole_frac=0.5, max_faces=3):
    """Face crops for one sampled frame. A frame whose largest face already fills `whole_frac` of the width is a
    face close-up and is used whole (same rule as the image UI); a frame with no face gives one whole-frame crop with
    track id -1. Returns [(track_id, crop PIL, box or None)]."""
    if frame.cut:
        tracker.reset()
    img = Image.fromarray(frame.image)
    dets = detect_faces(frame.image)[:max_faces]
    tracked = tracker.update(frame.t, dets)
    if not tracked:
        return [(-1, img, None)]
    if (dets[0][2] - dets[0][0]) >= whole_frac * img.size[0]:
        return [(tracked[0][0], img, None)]
    return [(tid, img.crop(crop_box(box, img.size)), box) for tid, box, _ in tracked]
