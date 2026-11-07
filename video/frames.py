"""Video reading and scene-aware frame sampling.

The image model sees single frames, so which frames we pick decides what a video verdict can notice: an even grid
alone misses a manipulated shot that is shorter than the grid step, and wastes frames inside long static shots. We
find shot cuts with a cheap histogram pre-pass and always include the first frames of every shot, then fill the rest
of the budget with an even grid.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self):
        return self.n_frames / self.fps if self.fps else 0.0


@dataclass
class Frame:
    idx: int          # frame index in the video
    t: float          # seconds
    scene: int        # shot number, 0-based
    cut: bool         # first sampled frame of a shot (face tracks restart here)
    image: np.ndarray = field(repr=False)    # RGB uint8 H x W x 3


def read_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {path}')
    info = VideoInfo(path, cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def _hist(bgr):
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)


def find_cuts(path, probe_fps=4.0, threshold=0.5):
    """Frame indices where a new shot starts (histogram distance between probe frames), always including 0.
    Reads sequentially (grab() skips decoding of frames that are not probed), so it works on containers that
    cannot seek reliably."""
    info = read_info(path)
    step = max(1, int(round(info.fps / probe_fps)))
    cap = cv2.VideoCapture(path)
    cuts, prev, i = [0], None, 0
    while True:
        if i % step == 0:
            ok, f = cap.read()
            if not ok:
                break
            h = _hist(f)
            if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > threshold:
                cuts.append(i)
            prev = h
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return cuts, i


def plan_indices(n_frames, cuts, max_frames, per_cut=2, cut_offset=2):
    """Which frame indices to decode: `per_cut` frames at the start of every shot (the first `cut_offset` frames after
    a cut are often blended), then an even grid for the remaining budget. Returns sorted unique indices."""
    n_frames = max(n_frames, 1)
    chosen = set()
    budget_cuts = max(1, max_frames // 2)
    for c in cuts[:budget_cuts]:
        for k in range(per_cut):
            chosen.add(min(n_frames - 1, c + cut_offset + k * 3))
    rest = max_frames - len(chosen)
    if rest > 0:
        chosen.update(np.linspace(0, n_frames - 1, num=min(rest, n_frames)).round().astype(int).tolist())
    idx = sorted(chosen)
    if len(idx) > max_frames:    # thin evenly, keeping the first and the last
        idx = [idx[j] for j in np.linspace(0, len(idx) - 1, num=max_frames).round().astype(int)]
    return sorted(set(idx))


def sample_frames(path, max_frames=32, scene_aware=True, probe_fps=4.0, cut_threshold=0.5):
    """Decode up to `max_frames` frames. Returns (VideoInfo, [Frame], cuts)."""
    info = read_info(path)
    if scene_aware:
        cuts, counted = find_cuts(path, probe_fps, cut_threshold)
        n = info.n_frames if info.n_frames > 0 else counted
    else:
        cuts, n = [0], info.n_frames
    idx = plan_indices(n, cuts, max_frames)
    cap = cv2.VideoCapture(path)
    frames, want, i = [], set(idx), 0
    last = max(idx)
    cut_arr = np.array(cuts)
    seen_scene = set()
    while i <= last:    # one sequential pass; seeking is unreliable on many codecs
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                scene = int(np.searchsorted(cut_arr, i, side='right') - 1)
                first = scene not in seen_scene
                seen_scene.add(scene)
                frames.append(Frame(i, i / info.fps, scene, first, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    return info, frames, cuts
