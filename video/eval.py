"""Part 5: evaluate the video pipeline on labelled held-out videos.

    python -m video.eval --root /path/to/ffpp_test --stagea runs/stagea_v7/last.pt --m2f2-shard <shard 3> --out-dir eval_video/
    python -m video.eval --manifest videos.csv --cache-dir eval_video/frames --out-dir eval_video/        # reuse cached Part 1 output

Videos come from a manifest (CSV with columns `video,label[,method,fake_start,fake_end]`, label 0/1 or real/fake) or from a
folder tree whose sub-folder names give the label and the method (`real/`, `original/`, `youtube/` ... are real; every other
folder, e.g. `Deepfakes/` or `Face2Face/`, is fake and its name is the method). Held-out data only: this module never trains or
tunes anything; the decision threshold is the fixed one of Part 2.

For every video the Part 1 records are cached (`<cache-dir>/<name>.frames.jsonl`), the Part 2 timeline gives the verdict, and the
report compares three video scores, so the benefit of the temporal part is measured, not assumed:
  mean      trimmed mean of the per-frame probabilities (the image-model baseline, the paper's protocol)
  max       the strongest smoothed per-frame probability
  segment   the Part 2 score: the strongest suspect segment's mean probability, else the trimmed mean
Metrics: AUC, accuracy, true / false positive rate and the share of 'uncertain' verdicts, overall and per method; localisation IoU
when the manifest gives the manipulated interval. Output: `report.json` and `report.md` next to the per-video `results.csv`.
"""
import argparse
import csv
import json
import os

import numpy as np

from video.timeline import build_timeline, load_records, smooth, group_tracks, trimmed_mean

REAL_NAMES = {'real', 'original', 'originals', 'original_sequences', 'youtube', 'pristine', 'authentic', '0'}
EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def parse_label(v):
    s = str(v).strip().lower()
    if s in ('1', 'fake', 'manipulated', 'true'):
        return 1
    if s in ('0', 'real', 'original', 'pristine', 'false'):
        return 0
    raise ValueError(f'label must be 0/1 or real/fake, got {v!r}')


def read_manifest(path):
    """[{video, label, method, fake_start, fake_end}] from a CSV with columns video,label[,method,fake_start,fake_end]."""
    rows = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            vid = r['video'] if os.path.isabs(r['video']) else os.path.join(base, r['video'])
            lab = parse_label(r['label'])
            rows.append({'video': vid, 'label': lab, 'method': (r.get('method') or ('real' if lab == 0 else 'fake')).strip(),
                         'fake_start': float(r['fake_start']) if r.get('fake_start') else None,
                         'fake_end': float(r['fake_end']) if r.get('fake_end') else None})
    return rows


def scan_root(root):
    """Videos under root; the first folder level below it names the label / method (deeper folders are ignored)."""
    rows = []
    for dp, _, fs in sorted(os.walk(root)):
        for fn in sorted(fs):
            if not fn.lower().endswith(EXTS):
                continue
            rel = os.path.relpath(os.path.join(dp, fn), root).split(os.sep)
            top = rel[0] if len(rel) > 1 else 'unknown'
            lab = 0 if top.lower() in REAL_NAMES else 1
            rows.append({'video': os.path.join(dp, fn), 'label': lab, 'method': 'real' if lab == 0 else top,
                         'fake_start': None, 'fake_end': None})
    if not rows:
        raise SystemExit(f'no videos ({", ".join(EXTS)}) under {root}')
    return rows


def auc(labels, scores):
    """ROC AUC by ranks (ties get the mean rank). None if one class is missing."""
    y, s = np.asarray(labels), np.asarray(scores, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    i = 0
    ss = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def video_scores(header, rows):
    """The three video-level scores of the report, from the Part 1 records of one video."""
    tl = build_timeline(header, rows)
    ps = [r['p_fake'] for r in rows]
    mean = trimmed_mean(ps) if ps else 0.0
    mx = 0.0
    for tr in group_tracks(rows).values():
        tr = sorted(tr, key=lambda r: r['t'])
        mx = max(mx, float(np.max(smooth([r['t'] for r in tr], [r['p_fake'] for r in tr]))))
    seg = max([s['mean_p'] for s in tl['segments']], default=None)
    return tl, {'mean': mean, 'max': mx, 'segment': seg if seg is not None else mean}


def interval_iou(pred, gt):
    """IoU of two sets of (start, end) intervals (each merged first)."""
    def merge(iv):
        out = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out
    p, g = merge(pred), merge(gt)
    ln = lambda iv: sum(b - a for a, b in iv)
    inter = sum(max(0.0, min(b, d) - max(a, c)) for a, b in p for c, d in g)
    union = ln(p) + ln(g) - inter
    return inter / union if union > 0 else 0.0


def evaluate_video(item, rows, header):
    tl, sc = video_scores(header, rows)
    v = tl['verdict']
    res = {'video': item['video'], 'label': item['label'], 'method': item['method'], 'verdict': v['label'],
           'p_video': round(v['p_video'], 4), 'score_mean': round(sc['mean'], 4), 'score_max': round(sc['max'], 4),
           'score_segment': round(sc['segment'], 4), 'n_segments': len(tl['segments']), 'dominant_type': v.get('dominant_type'),
           'iou': None}
    if item.get('fake_start') is not None and item.get('fake_end') is not None:
        res['iou'] = round(interval_iou([(s['start'], s['end']) for s in tl['segments']],
                                        [(item['fake_start'], item['fake_end'])]), 4)
    return res


def _rates(res):
    """Verdict statistics of a group of per-video results; 'uncertain' is counted as neither fake nor real."""
    n = len(res)
    if n == 0:
        return None
    fake_pred = sum(r['verdict'] == 'fake' for r in res)
    unc = sum(r['verdict'] == 'uncertain' for r in res)
    pos = [r for r in res if r['label'] == 1]
    neg = [r for r in res if r['label'] == 0]
    correct = sum((r['verdict'] == 'fake') == (r['label'] == 1) and r['verdict'] != 'uncertain' for r in res)
    out = {'n': n, 'accuracy': correct / n, 'uncertain': unc / n, 'predicted_fake': fake_pred / n}
    if pos:
        out['tpr'] = sum(r['verdict'] == 'fake' for r in pos) / len(pos)
    if neg:
        out['fpr'] = sum(r['verdict'] == 'fake' for r in neg) / len(neg)
    return out


def summarize(results):
    labels = [r['label'] for r in results]
    rep = {'n_videos': len(results), 'n_real': labels.count(0), 'n_fake': labels.count(1),
           'auc': {k: auc(labels, [r['score_' + k] for r in results]) for k in ('mean', 'max', 'segment')},
           'overall': _rates(results), 'by_method': {}}
    for m in sorted({r['method'] for r in results if r['label'] == 1}):
        sub = [r for r in results if r['method'] == m] + [r for r in results if r['label'] == 0]
        rep['by_method'][m] = {'n_fake': sum(r['method'] == m for r in results),
                               'recall': _rates([r for r in results if r['method'] == m]).get('tpr'),
                               'auc_segment': auc([r['label'] for r in sub], [r['score_segment'] for r in sub])}
    ious = [r['iou'] for r in results if r['iou'] is not None and r['label'] == 1]
    if ious:
        rep['localisation'] = {'n': len(ious), 'mean_iou': float(np.mean(ious))}
    return rep


def _f(x):
    return 'n/a' if x is None else f'{x:.3f}'


def markdown(rep):
    o = rep['overall']
    L = [f"# Video evaluation\n\n{rep['n_videos']} videos ({rep['n_real']} real, {rep['n_fake']} fake).\n",
         '| video score | AUC |', '|---|---|']
    L += [f'| {k} | {_f(v)} |' for k, v in rep['auc'].items()]
    L += ['', f"Verdicts (Part 2 rules): accuracy {_f(o['accuracy'])}, fake recall {_f(o.get('tpr'))}, "
          f"false positive rate {_f(o.get('fpr'))}, uncertain {_f(o['uncertain'])}.", '',
          '| method | fake videos | recall | AUC vs real |', '|---|---|---|---|']
    L += [f"| {m} | {d['n_fake']} | {_f(d['recall'])} | {_f(d['auc_segment'])} |" for m, d in rep['by_method'].items()]
    if 'localisation' in rep:
        L += ['', f"Localisation: mean IoU {_f(rep['localisation']['mean_iou'])} over {rep['localisation']['n']} videos with a known interval."]
    return '\n'.join(L) + '\n'


def write_outputs(results, rep, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = list(results[0].keys())
    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write(markdown(rep))


def cache_path(cache_dir, video):
    stem = os.path.splitext(os.path.basename(video))[0]
    tag = os.path.basename(os.path.dirname(os.path.abspath(video)))
    return os.path.join(cache_dir, f'{tag}__{stem}.frames.jsonl')


def get_records(item, cache_dir, engine, max_frames):
    """Part 1 output of a video: from the cache if present, else computed (needs the engine) and cached."""
    cp = cache_path(cache_dir, item['video'])
    if os.path.exists(cp):
        return load_records(cp)
    if engine is None:
        raise FileNotFoundError(f'no cached records for {item["video"]} ({cp}) and no model given (--stagea / --m2f2-shard)')
    from video.analyze import analyze
    info, cuts, rows = analyze(item['video'], engine, max_frames)
    header = {'video': info.path, 'fps': info.fps, 'n_frames': info.n_frames, 'size': [info.width, info.height],
              'duration': round(info.duration, 3), 'cuts': cuts}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cp, 'w') as f:
        f.write(json.dumps(header) + '\n')
        f.writelines(json.dumps(r) + '\n' for r in rows)
    return header, rows


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--manifest')
    g.add_argument('--root')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--cache-dir', default=None, help='default: <out-dir>/frames')
    ap.add_argument('--stagea')
    ap.add_argument('--m2f2-shard')
    ap.add_argument('--max-frames', type=int, default=32)
    ap.add_argument('--device', default=None)
    ap.add_argument('--limit', type=int, default=None, help='first N videos per class (quick check)')
    a = ap.parse_args()
    items = read_manifest(a.manifest) if a.manifest else scan_root(a.root)
    if a.limit:
        keep = []
        for lab in (0, 1):
            keep += [i for i in items if i['label'] == lab][:a.limit]
        items = keep
    cache = a.cache_dir or os.path.join(a.out_dir, 'frames')
    engine = None
    if a.stagea and a.m2f2_shard:
        from app.forensic_engine import ForensicEngine
        engine = ForensicEngine(a.stagea, a.m2f2_shard, device=a.device)
    results = []
    for k, it in enumerate(items, 1):
        try:
            header, rows = get_records(it, cache, engine, a.max_frames)
            results.append(evaluate_video(it, rows, header))
        except Exception as e:                       # one unreadable video must not end a long run; it is reported, not hidden
            print(f'[{k}/{len(items)}] SKIPPED {it["video"]}: {e}')
            continue
        print(f'[{k}/{len(items)}] {os.path.basename(it["video"])} label={it["label"]} -> {results[-1]["verdict"]} '
              f'(p {results[-1]["p_video"]})')
    if not results:
        raise SystemExit('no video could be evaluated')
    rep = summarize(results)
    rep['skipped'] = len(items) - len(results)
    write_outputs(results, rep, a.out_dir)
    print(markdown(rep))


if __name__ == '__main__':
    main()
