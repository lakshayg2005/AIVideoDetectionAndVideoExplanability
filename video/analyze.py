"""Part 1: per-frame forensic evidence for a video.

    python -m video.analyze clip.mp4 --stagea runs/stagea_v7/last.pt --m2f2-shard <shard 3> --out clip.frames.jsonl

One JSON line per (sampled frame, face track): time, shot, track id, face box, fake probability, edit-type and filter
probabilities and the share of the face crop that the forgery heatmap marks. Later parts turn this into a timeline,
overlays and an explanation.
"""
import argparse
import json
import os

import numpy as np

from video.faces import face_crops
from video.frames import sample_frames


def record(frame, tid, box, r):
    types = r['types']
    top = max(types, key=types.get)
    return {'t': round(frame.t, 3), 'frame': frame.idx, 'scene': frame.scene, 'cut': frame.cut, 'track': tid,
            'box': [round(float(v), 1) for v in box] if box is not None else None,
            'p_fake': round(r['p_fake'], 4), 'type': top, 'types': {k: round(v, 4) for k, v in types.items()},
            'filters': {k: round(v, 4) for k, v in r['filters'].items()},
            'forgery_area': round(float((r['forgery_map'] > 0.5).mean()), 4)}


def analyze(path, engine, max_frames=32, scene_aware=True, max_faces=3):
    """Returns (VideoInfo, cuts, [record])."""
    from video.faces import FaceTracker
    info, frames, cuts = sample_frames(path, max_frames, scene_aware)
    tracker, rows = FaceTracker(), []
    for fr in frames:
        for tid, crop, box in face_crops(fr, tracker, max_faces=max_faces):
            rows.append(record(fr, tid, box, engine.analyze(crop)))
    return info, cuts, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--stagea', required=True, help='forensics.train checkpoint (last.pt)')
    ap.add_argument('--m2f2-shard', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-frames', type=int, default=32)
    ap.add_argument('--no-scene-aware', action='store_true')
    ap.add_argument('--device', default=None, help='cpu / cuda (default: cuda if available)')
    a = ap.parse_args()
    from app.forensic_engine import ForensicEngine
    engine = ForensicEngine(a.stagea, a.m2f2_shard, device=a.device)
    info, cuts, rows = analyze(a.video, engine, a.max_frames, not a.no_scene_aware)
    out = a.out or os.path.splitext(a.video)[0] + '.frames.jsonl'
    with open(out, 'w') as f:
        f.write(json.dumps({'video': info.path, 'fps': info.fps, 'n_frames': info.n_frames, 'size': [info.width, info.height],
                            'duration': round(info.duration, 3), 'cuts': cuts}) + '\n')
        for r in rows:
            f.write(json.dumps(r) + '\n')
    p = [r['p_fake'] for r in rows]
    print(f'{len(rows)} records from {len(set(r["frame"] for r in rows))} frames, {len(cuts)} shot(s) -> {out}; '
          f'p_fake mean {np.mean(p):.3f} max {np.max(p):.3f}')


if __name__ == '__main__':
    main()
