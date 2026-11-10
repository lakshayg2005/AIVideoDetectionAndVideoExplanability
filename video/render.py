"""Part 3: what a person looks at - a timeline plot, an annotated video and a strip of key frames with heatmaps.

    python -m video.render clip.mp4 clip.frames.jsonl clip.timeline.json --out-dir out/ \
        [--stagea runs/stagea_v7/last.pt --m2f2-shard <model-00003-of-00003.safetensors>]

Writes <stem>.timeline.png (fake probability per face over time, suspect segments shaded, shot cuts marked),
<stem>.annotated.mp4 (every frame: the face box coloured by the interpolated fake probability, plus a verdict bar with the
suspect segments and a playhead) and <stem>.keyframes.png (the most telling frames: the peak frame of every segment and the
cleanest frames, each with the forgery heatmap of its face crop). The heatmaps need the model (--stagea / --m2f2-shard);
without it the strip shows the crops only. Everything else uses the Part 1 / Part 2 files alone.
"""
import argparse
import json
import os

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from video.faces import crop_box  # noqa: E402
from video.timeline import GAP_S, THR, load_records  # noqa: E402

BAR_H = 36


def p_color(p):
    """Green (clean) -> amber (unsure) -> red (fake), as RGB ints."""
    g, a, r = np.array([60, 190, 90]), np.array([240, 170, 40]), np.array([230, 50, 50])
    p = float(np.clip(p, 0, 1))
    c = g + (a - g) * (p / 0.5) if p < 0.5 else a + (r - a) * ((p - 0.5) / 0.5)
    return tuple(int(x) for x in c)


def track_name(r):
    return f"track {r['track']}" if r['track'] >= 0 else f"no-face shot {r['scene']}"


# ------------------------------------------------------------------------------------------------ timeline plot
def timeline_plot(header, timeline, out_png):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cmap = plt.get_cmap('tab10')
    for i, (name, pts) in enumerate(sorted(timeline['timeline'].items())):
        t = [x['t'] for x in pts]
        ax.scatter(t, [x['p_fake'] for x in pts], s=14, color=cmap(i % 10), alpha=0.45)
        ax.plot(t, [x['smoothed'] for x in pts], color=cmap(i % 10), lw=1.8, label=name)
    thr = timeline['params']['threshold']
    ax.axhline(thr, ls='--', color='grey', lw=1)
    for k, s in enumerate(timeline['segments']):
        ax.axvspan(s['start'] - 0.15, s['end'] + 0.15, color='#d1495b', alpha=0.15, label='suspect segment' if k == 0 else None)
        ax.text((s['start'] + s['end']) / 2, 1.03, f"{s['type'] or ''}\n{s['mean_p']:.2f}", ha='center', va='bottom', fontsize=7)
    fps = header.get('fps') or 25.0
    for c in header.get('cuts', [])[1:]:
        ax.axvline(c / fps, ls=':', color='k', lw=0.9)
    v = timeline['verdict']
    why = '  (localized segment)' if v['label'] == 'fake' and v['p_video'] < thr else ''
    ax.set(ylim=(-0.02, 1.15), xlim=(0, max(header.get('duration') or 0, max(x['t'] for p in timeline['timeline'].values() for x in p) + 0.5)),
           xlabel='time (s)', ylabel='fake probability',
           title=f"{v['label'].upper()}{why}   video score {v['p_video']:.2f}   (dots: single frames, lines: smoothed; dotted: cuts)")
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    return out_png


# ------------------------------------------------------------------------------------------------ annotated video
def _label(f, text, org, col, scale=0.7):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(f, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(f, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


def _curves(timeline, rows):
    """Per track: sample times, smoothed p and face boxes (None when the whole frame was analysed)."""
    box = {(track_name(r), r['frame']): r['box'] for r in rows}
    out = {}
    for name, pts in timeline['timeline'].items():
        out[name] = dict(t=np.array([x['t'] for x in pts]), p=np.array([x['smoothed'] for x in pts]),
                         box=[box.get((name, x['frame'])) for x in pts])
    return out


def _at(curve, tt, gap_s=GAP_S, hold_s=0.5):
    """Interpolated (p, box) of one track at time tt, or None when no sample is close enough."""
    t = curve['t']
    if len(t) == 0 or tt < t[0] - hold_s or tt > t[-1] + hold_s:
        return None
    j = int(np.searchsorted(t, tt))
    if j == 0 or j >= len(t):
        i = 0 if j == 0 else len(t) - 1
        return float(curve['p'][i]), curve['box'][i]
    i = j - 1
    if t[j] - t[i] > gap_s:
        k = i if tt - t[i] <= t[j] - tt else j
        return (float(curve['p'][k]), curve['box'][k]) if abs(tt - t[k]) <= hold_s else None
    w = (tt - t[i]) / max(t[j] - t[i], 1e-9)
    p = float(curve['p'][i] * (1 - w) + curve['p'][j] * w)
    bi, bj = curve['box'][i], curve['box'][j]
    box = [bi[m] * (1 - w) + bj[m] * w for m in range(4)] if bi is not None and bj is not None else (bi if w < 0.5 else bj)
    return p, box


def _bar(width, duration, curves, segments):
    """Verdict bar image [BAR_H, width, 3] (RGB): colour = highest smoothed p over all tracks at that time."""
    xs = np.arange(width) / max(width - 1, 1) * duration
    best = np.full(width, np.nan)
    for c in curves.values():
        if len(c['t']):
            v = np.interp(xs, c['t'], c['p'], left=np.nan, right=np.nan)
            best = np.fmax(best, v)
    bar = np.zeros((BAR_H, width, 3), np.uint8) + 35
    for x in range(width):
        if not np.isnan(best[x]):
            bar[4:BAR_H - 4, x] = p_color(best[x])
    for s in segments:
        x0, x1 = int(s['start'] / duration * (width - 1)), max(int(s['end'] / duration * (width - 1)), int(s['start'] / duration * (width - 1)) + 2)
        bar[0:3, x0:x1] = (255, 255, 255)
    return bar


def annotate_video(video, header, rows, timeline, out_mp4):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or header.get('fps') or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = header.get('duration') or n / fps
    curves = _curves(timeline, rows)
    bar = _bar(W, duration, curves, timeline['segments'])
    wr = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H + BAR_H))
    i, written = 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        tt = i / fps
        for name, c in curves.items():
            r = _at(c, tt)
            if r is None:
                continue
            p, box = r
            col = p_color(p)[::-1]    # BGR
            if box is not None:
                x0, y0, x1, y1 = [int(v) for v in box]
                cv2.rectangle(f, (x0, y0), (x1, y1), col, 3)
                _label(f, f'{p:.2f}', (x0, max(y0 - 8, 18)), col)
            else:    # whole frame analysed: a coloured frame border and the score in the corner
                cv2.rectangle(f, (0, 0), (W - 1, H - 1), col, 6)
                _label(f, f'{p:.2f}', (12, 32), col, 0.9)
        b = bar.copy()
        x = int(np.clip(tt / duration, 0, 1) * (W - 1))
        b[:, max(x - 1, 0):x + 2] = 255
        wr.write(np.vstack([f, cv2.cvtColor(b, cv2.COLOR_RGB2BGR)]))
        i += 1
        written += 1
    cap.release()
    wr.release()
    return written


# ------------------------------------------------------------------------------------------------ key frames
def select_key_records(rows, timeline, n=6):
    """Peak frame of every suspect segment (most probable first if there are more than n) + the cleanest frames."""
    by = {(track_name(r), r['frame']): r for r in rows}
    peaks = []
    for s in timeline['segments']:
        cand = [by[(s['track'], f)] for f in s['frames'] if (s['track'], f) in by]
        if cand:
            peaks.append(max(cand, key=lambda r: r['p_fake']))
    peaks = sorted(peaks, key=lambda r: -r['p_fake'])[:n]
    used = {(track_name(r), r['frame']) for r in peaks}
    seg_frames = {(s['track'], f) for s in timeline['segments'] for f in s['frames']}
    clean = sorted((r for r in rows if (track_name(r), r['frame']) not in seg_frames and (track_name(r), r['frame']) not in used),
                   key=lambda r: r['p_fake'])
    take = clean[:max(0, min(2, n - len(peaks)))] if peaks else clean[:min(3, n)]
    return sorted(peaks + take, key=lambda r: r['t'])


def read_frames(video, indices):
    """{frame index: RGB array} in one sequential pass (seeking is unreliable on many codecs)."""
    want, out = set(indices), {}
    cap = cv2.VideoCapture(video)
    i, last = 0, max(indices) if indices else -1
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def make_tile(frame, rec, forgery_map=None, height=300):
    img = Image.fromarray(frame)
    col = p_color(rec['p_fake'])
    d = ImageDraw.Draw(img)
    box = rec['box']
    if box is not None:
        d.rectangle([float(v) for v in box], outline=col, width=max(3, img.size[1] // 150))
    crop = img.crop(crop_box(box, img.size)) if box is not None else Image.fromarray(frame)
    if forgery_map is not None:
        from app.forensic_engine import draw_heatmap    # heavy imports (torch) only when heatmaps are wanted
        crop = draw_heatmap(crop, forgery_map)
    fit = lambda im: im.resize((max(1, round(im.size[0] * height / im.size[1])), height), Image.BILINEAR)
    left, right = fit(img), fit(crop)
    tile = Image.new('RGB', (left.size[0] + right.size[0] + 10, height + 26), (30, 30, 30))
    tile.paste(left, (0, 26))
    tile.paste(right, (left.size[0] + 10, 26))
    ImageDraw.Draw(tile).text((6, 6), f"t = {rec['t']:.1f} s   p(fake) = {rec['p_fake']:.2f}   type: {rec['type']}   "
                                      f"marked area: {100 * rec['forgery_area']:.0f}%", fill=col)
    return tile


def keyframe_strip(video, rows, timeline, out_png, engine=None, n=6, height=300):
    recs = select_key_records(rows, timeline, n)
    frames = read_frames(video, [r['frame'] for r in recs])
    tiles = []
    for r in recs:
        if r['frame'] not in frames:
            continue
        fmap = None
        if engine is not None:
            crop = Image.fromarray(frames[r['frame']])
            if r['box'] is not None:
                crop = crop.crop(crop_box(r['box'], crop.size))
            fmap = engine.analyze(crop)['forgery_map']
        tiles.append(make_tile(frames[r['frame']], r, fmap, height))
    if not tiles:
        return None
    W = max(t.size[0] for t in tiles)
    strip = Image.new('RGB', (W, sum(t.size[1] + 6 for t in tiles)), (255, 255, 255))
    y = 0
    for t in tiles:
        strip.paste(t, (0, y))
        y += t.size[1] + 6
    strip.save(out_png)
    return out_png, [r['frame'] for r in recs]


def render_all(video, frames_jsonl, timeline_json, out_dir, engine=None, n_key=6, make_video=True):
    header, rows = load_records(frames_jsonl)
    timeline = json.load(open(timeline_json))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(video))[0])
    out = {'timeline_png': timeline_plot(header, timeline, stem + '.timeline.png')}
    kf = keyframe_strip(video, rows, timeline, stem + '.keyframes.png', engine, n_key)
    out['keyframes_png'], out['keyframe_frames'] = kf if kf else (None, [])
    if make_video:
        out['annotated_frames'] = annotate_video(video, header, rows, timeline, stem + '.annotated.mp4')
        out['annotated_mp4'] = stem + '.annotated.mp4'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('frames_jsonl')
    ap.add_argument('timeline_json')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--stagea', default=None, help='forensics.train checkpoint: adds the forgery heatmaps to the key frames')
    ap.add_argument('--m2f2-shard', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--n-key', type=int, default=6)
    ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args()
    engine = None
    if a.stagea:
        if not a.m2f2_shard:
            ap.error('--stagea needs --m2f2-shard')
        from app.forensic_engine import ForensicEngine
        engine = ForensicEngine(a.stagea, a.m2f2_shard, device=a.device)
    res = render_all(a.video, a.frames_jsonl, a.timeline_json, a.out_dir, engine, a.n_key, not a.no_video)
    for k, v in res.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
