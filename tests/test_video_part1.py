"""Part 1 checks on a synthetic 2-shot video (a real face image moving over a plain background; shot 2 has another
background colour). Run: python tests/test_video_part1.py [face_image.png]. No model needed."""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.faces import FaceTracker, face_crops, iou
from video.frames import find_cuts, plan_indices, read_info, sample_frames

FACE = sys.argv[1] if len(sys.argv) > 1 else '/data3/Aditya/bp_data/data/DiffusionFace/x/Real/0.png'


def make_video(path, fps=25, secs=(4, 4), size=(640, 360)):
    face = cv2.resize(cv2.imread(FACE), (180, 180))
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    n = 0
    for shot, (s, color) in enumerate(zip(secs, [(200, 120, 40), (30, 60, 200)])):
        for k in range(int(s * fps)):
            f = np.full((size[1], size[0], 3), color, np.uint8)
            f[::20] = 255 - np.array(color)    # stripes so the histograms differ clearly between shots
            x = 60 + int(300 * (k / (s * fps)))
            f[90:270, x:x + 180] = face
            w.write(f)
            n += 1
    w.release()
    return n


def main():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'two_shots.mp4')
    n = make_video(p)
    info = read_info(p)
    assert info.n_frames == n and abs(info.fps - 25) < 1e-3 and (info.width, info.height) == (640, 360), info
    cuts, _ = find_cuts(p)
    assert len(cuts) == 2 and abs(cuts[1] - 100) <= 6, cuts    # shot 2 starts at frame 100
    idx = plan_indices(n, cuts, 12)
    assert len(idx) <= 12 and any(100 <= i <= 110 for i in idx) and idx[0] <= 4 and idx[-1] >= n - 2, idx
    info, frames, cuts = sample_frames(p, 12)
    assert len(frames) == len(set(f.idx for f in frames)) and [f.idx for f in frames] == sorted(f.idx for f in frames)
    assert sum(f.cut for f in frames) == 2 and {f.scene for f in frames} == {0, 1}
    tr, ids = FaceTracker(), []
    for fr in frames:
        c = face_crops(fr, tr)
        assert len(c) == 1
        ids.append((fr.scene, c[0][0]))
    assert all(t >= 0 for _, t in ids), f'face not found in some frames: {ids}'
    for s in (0, 1):    # one id per shot, and tracks restart at the cut
        assert len({t for sc, t in ids if sc == s}) == 1, ids
    assert ids[0][1] != ids[-1][1], ids
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    print(f'OK  {len(frames)} frames, cuts {cuts}, track ids per frame {[t for _, t in ids]}')


if __name__ == '__main__':
    main()
