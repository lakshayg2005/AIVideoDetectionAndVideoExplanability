"""Part 3 checks: timeline plot, annotated video, key frames. Synthetic 8 s two-shot video (a face image moving over a plain
background) with hand-made Part 1 records (clean shot, then a fake shot); a stub engine stands in for the model.
Run: python tests/test_video_part3.py   (needs the face image used by test_video_part1)"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_video_part1 import make_video          # noqa: E402
from video.render import BAR_H, render_all, select_key_records, track_name   # noqa: E402
from video.timeline import build_timeline        # noqa: E402

TYPES = ['real', 'face_swap', 'splice', 'diffusion_local', 'img2img', 'generated']


def rec(t, p, scene, track):
    k = t if t < 4 else t - 4                     # position inside its shot (4 s each)
    x = 60 + int(300 * (k / 4))
    types = {k_: 0.0 for k_ in TYPES}
    types['face_swap' if p >= 0.5 else 'real'] = 1.0
    return dict(t=float(t), frame=int(t * 25), scene=scene, cut=(t in (0, 4)), track=track, box=[x, 90, x + 180, 270], p_fake=p,
                type='face_swap' if p >= 0.5 else 'real', types=types, filters={'blur': 0.1}, forgery_area=0.3 if p >= .5 else 0.0)


class StubEngine:
    def analyze(self, img):
        m = np.zeros((img.size[1], img.size[0]), np.float32)
        m[m.shape[0] // 4:m.shape[0] // 2, m.shape[1] // 4:m.shape[1] // 2] = 0.9
        return {'forgery_map': m}


def frame_at(path, i):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    cap.release()
    assert ok, f'cannot read frame {i}'
    return f


def count(f, rgb, tol=70):
    d = np.abs(f[..., ::-1].astype(int) - np.array(rgb)[None, None]).sum(-1)
    return int((d < tol).sum())


def main():
    d = tempfile.mkdtemp()
    v = os.path.join(d, 'clip.mp4')
    n = make_video(v)
    rows = [rec(t, 0.05, 0, 0) for t in range(0, 4)] + [rec(t, p, 1, 1) for t, p in zip(range(4, 8), (0.92, 0.95, 0.9, 0.94))]
    header = dict(video=v, fps=25.0, duration=8.0, cuts=[0, 100])
    tl = build_timeline(header, rows)
    assert len(tl['segments']) == 1 and tl['verdict']['label'] == 'fake'
    fj, tj = os.path.join(d, 'clip.frames.jsonl'), os.path.join(d, 'clip.timeline.json')
    with open(fj, 'w') as f:
        f.write(json.dumps(header) + '\n' + '\n'.join(json.dumps(r) for r in rows) + '\n')
    json.dump(tl, open(tj, 'w'))

    # key frames: the segment's peak frame (p = 0.95 at t = 5 s) and a clean frame from the first shot
    keys = select_key_records(rows, tl, n=6)
    assert any(r['frame'] == 125 for r in keys), [r['frame'] for r in keys]
    assert any(r['p_fake'] < 0.5 for r in keys) and 2 <= len(keys) <= 4, [r['frame'] for r in keys]
    clean_only = build_timeline(header, [rec(t, 0.05, 0, 0) for t in range(4)])
    assert 1 <= len(select_key_records([rec(t, 0.05, 0, 0) for t in range(4)], clean_only, n=6)) <= 3

    out = render_all(v, fj, tj, os.path.join(d, 'out'), engine=StubEngine(), n_key=6)
    im = Image.open(out['timeline_png'])
    assert im.size[0] > 800 and im.size[1] > 300, im.size
    strip = Image.open(out['keyframes_png'])
    assert strip.size[1] > 2 * 300 and strip.size[0] > 600, strip.size
    assert set(out['keyframe_frames']) == {r['frame'] for r in keys}

    # annotated video: same frames, bar added, red box in the fake shot, green box in the clean one, bar colour follows
    cap = cv2.VideoCapture(out['annotated_mp4'])
    w, h, nf = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    cap.release()
    assert (w, h) == (640, 360 + BAR_H) and abs(nf - n) <= 1 and out['annotated_frames'] == n, (w, h, nf, n)
    fake_f, clean_f = frame_at(out['annotated_mp4'], 150), frame_at(out['annotated_mp4'], 25)   # t = 6 s, t = 1 s
    red, green = (230, 50, 50), (60, 190, 90)
    assert count(fake_f[:360], red) > 80 and count(clean_f[:360], green) > 80, (count(fake_f[:360], red), count(clean_f[:360], green))
    assert count(clean_f[:360], red) < 30
    barrow = lambda f, t: f[360 + BAR_H // 2, int(t / 8 * 639)][::-1].astype(int)
    assert barrow(fake_f, 6)[0] > barrow(fake_f, 6)[1] and barrow(fake_f, 1)[1] > barrow(fake_f, 1)[0], (barrow(fake_f, 6), barrow(fake_f, 1))
    print(f'OK  plot {im.size}, strip {strip.size}, video {w}x{h} {nf} frames, key frames {out["keyframe_frames"]}')


if __name__ == '__main__':
    main()
