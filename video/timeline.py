"""Part 2: from per-frame evidence to a timeline, suspect segments and a video verdict.

    python -m video.timeline clip.frames.jsonl --out clip.timeline.json

Input is the output of video.analyze (header line + one record per sampled frame and face track). Frames are sampled
sparsely and a single frame can be wrong, so the scores of every face track are smoothed over time (a Gaussian whose width
adapts to how far apart the samples are), runs of smoothed scores above the threshold become *segments*, and the video
verdict combines the paper's trimmed-mean protocol with segment evidence, so a short manipulated shot is not averaged away
by a long untouched one. Pure numpy; no model needed.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

THR = 0.5            # fake threshold on a (smoothed) probability
SIGMA_MIN_S = 0.4    # smallest smoothing width in seconds
GAP_S = 1.5          # samples further apart than this never belong to one segment
MIN_FRAMES = 2       # a segment needs this many samples: the image heads flag some real faces, so one isolated frame is noise
SURE_P = 0.9         # an isolated sample this confident is not a segment but is listed as a spike for a human to look at
STRONG_P = 0.8       # a localized segment this convincing makes the whole video fake
STRONG_MIN_S, STRONG_MIN_FRAMES = 1.0, 3
UNSURE_BAND = (0.35, 0.65)


def load_records(path):
    """(header dict, [record dict]) from a video.analyze JSONL file."""
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if not lines or 'p_fake' in lines[0]:
        raise ValueError(f'{path}: first line must be the video header written by video.analyze')
    return lines[0], lines[1:]


def trimmed_mean(xs, trim=0.1):
    """Mean after dropping the lowest and highest `trim` share (the paper's video protocol)."""
    s = sorted(xs)
    k = int(len(s) * trim)
    if len(s) - 2 * k >= 1:
        s = s[k:len(s) - k]
    return float(np.mean(s))


def smooth(times, p, sigma_min=SIGMA_MIN_S):
    """Gaussian smoothing over (irregular) sample times; the width follows the median spacing so sparse sampling is not
    smoothed less than dense sampling. Returns an array the length of p."""
    t, p = np.asarray(times, float), np.asarray(p, float)
    if len(t) < 2:
        return p.copy()
    sigma = max(sigma_min, 0.75 * float(np.median(np.diff(t))))
    w = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma) ** 2)
    return (w * p[None, :]).sum(1) / w.sum(1)


def group_tracks(rows):
    """Records by track. Track ids are unique per video; track -1 (no face found, whole frame) is kept apart per shot."""
    g = defaultdict(list)
    for r in rows:
        g[(-1, r['scene']) if r['track'] < 0 else (r['track'], None)].append(r)
    return {k: sorted(v, key=lambda r: r['t']) for k, v in g.items()}


def find_segments(rows, sm, thr=THR, gap_s=GAP_S, min_frames=MIN_FRAMES):
    """Runs of consecutive suspect samples (smoothed p >= thr, neighbours not further apart than gap_s)."""
    runs, cur = [], []
    for r, s in zip(rows, sm):
        if s >= thr and (not cur or r['t'] - cur[-1][0]['t'] <= gap_s):
            cur.append((r, s))
            continue
        if cur:
            runs.append(cur)
        cur = [(r, s)] if s >= thr else []
    if cur:
        runs.append(cur)
    segs = []
    for run in runs:
        raw = [r['p_fake'] for r, _ in run]
        if len(run) < min_frames:
            continue
        fake_types = Counter()
        for r, _ in run:
            for k, v in r['types'].items():
                if k != 'real':
                    fake_types[k] += v
        segs.append(dict(start=run[0][0]['t'], end=run[-1][0]['t'], duration=round(run[-1][0]['t'] - run[0][0]['t'], 3),
                         n_frames=len(run), mean_p=float(np.mean(raw)), peak_p=float(max(raw)),
                         mean_smoothed_p=float(np.mean([s for _, s in run])),
                         type=fake_types.most_common(1)[0][0] if fake_types else None,
                         forgery_area=float(np.mean([r['forgery_area'] for r, _ in run])),
                         frames=[r['frame'] for r, _ in run]))
    return segs


def build_timeline(header, rows, thr=THR, sigma_min=SIGMA_MIN_S, gap_s=GAP_S):
    """The full Part 2 result as a JSON-serialisable dict."""
    if not rows:
        raise ValueError('no frame records')
    tracks, segments, tl = group_tracks(rows), [], {}
    for key, rs in tracks.items():
        sm = smooth([r['t'] for r in rs], [r['p_fake'] for r in rs], sigma_min)
        name = f'track {key[0]}' if key[0] >= 0 else f'no-face shot {key[1]}'
        tl[name] = [dict(t=r['t'], frame=r['frame'], p_fake=r['p_fake'], smoothed=round(float(s), 4)) for r, s in zip(rs, sm)]
        for sg in find_segments(rs, sm, thr, gap_s):
            segments.append(dict(sg, track=name))
    segments.sort(key=lambda s: s['start'])
    in_seg = {(s['track'], f) for s in segments for f in s['frames']}
    spikes = [dict(t=r['t'], frame=r['frame'], p_fake=r['p_fake'], track=r['track']) for r in rows
              if r['p_fake'] >= SURE_P and (f"track {r['track']}" if r['track'] >= 0 else f"no-face shot {r['scene']}", r['frame']) not in in_seg]

    p_all = [r['p_fake'] for r in rows]
    score = trimmed_mean(p_all)
    strong = [s for s in segments if s['mean_p'] >= STRONG_P and (s['duration'] >= STRONG_MIN_S or s['n_frames'] >= STRONG_MIN_FRAMES)]
    reasons = []
    if score >= thr:
        label = 'fake'
        reasons.append(f'trimmed-mean fake probability {score:.2f} >= {thr}')
    elif strong:
        label = 'fake'
        s = strong[0]
        reasons.append(f'a localized manipulation: {s["track"]}, {s["start"]:.1f}-{s["end"]:.1f} s, mean p {s["mean_p"]:.2f} over '
                       f'{s["n_frames"]} samples, although the overall trimmed mean is only {score:.2f}')
    elif UNSURE_BAND[0] <= score < UNSURE_BAND[1] or segments:
        label = 'uncertain'
        reasons.append(f'trimmed-mean {score:.2f}' + (f' with {len(segments)} weak suspect segment(s)' if segments else ' is close to the threshold'))
    else:
        label = 'real'
        reasons.append(f'trimmed-mean fake probability {score:.2f} and no suspect segment')

    suspect = [r for r in rows if r['p_fake'] >= thr]
    types = Counter()
    for r in suspect:
        for k, v in r['types'].items():
            if k != 'real':
                types[k] += v
    filt = {k: float(np.mean([r['filters'][k] for r in rows])) for k in rows[0]['filters']}
    return dict(video=header.get('video'), duration=header.get('duration'), n_samples=len(rows), n_tracks=len(tracks),
                verdict=dict(label=label, p_video=score, fake_fraction=float(np.mean(np.array(p_all) >= thr)),
                             dominant_type=types.most_common(1)[0][0] if types else None, reasons=reasons),
                segments=segments, isolated_spikes=spikes, timeline=tl, mean_filter_probs=filt,
                params=dict(threshold=thr, sigma_min_s=sigma_min, gap_s=gap_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_jsonl')
    ap.add_argument('--out', default=None)
    ap.add_argument('--threshold', type=float, default=THR)
    a = ap.parse_args()
    header, rows = load_records(a.frames_jsonl)
    res = build_timeline(header, rows, a.threshold)
    out = a.out or a.frames_jsonl.replace('.frames.jsonl', '') + '.timeline.json'
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    v = res['verdict']
    print(f"{v['label'].upper()}  p_video {v['p_video']:.3f}  fake samples {v['fake_fraction']:.0%}  type {v['dominant_type']}  "
          f"segments {len(res['segments'])}  -> {out}")
    if res['isolated_spikes']:
        print(f"  {len(res['isolated_spikes'])} isolated high-score sample(s) not counted as segments: "
              + ', '.join(f"{x['t']:.1f}s (p {x['p_fake']:.2f})" for x in res['isolated_spikes']))
    for s in res['segments']:
        print(f"  {s['track']}: {s['start']:.1f}-{s['end']:.1f} s  p {s['mean_p']:.2f}  {s['type']}  ({s['n_frames']} samples)")


if __name__ == '__main__':
    main()
