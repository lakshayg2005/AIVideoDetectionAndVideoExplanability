"""SPLIT - training-free detection of AI-generated and partially edited video (Hyun & Kim, ECCV 2026, arXiv 2607.02886).

A frozen vision encoder gives patch tokens P[t, n, :] for every frame. Two statistics on the token trajectories, no learned parameters:
  TTR  (two-step temporal roughness)  per patch: log2(L1 + eps) - log2(L2 + eps), L1 = path length of the trajectory,
                                      L2 = normalised length of its two-step chords; averaged over patches.
  LSMI (local spatial motion incoherence)  mean norm of the spatial gradients of the motion field M_t = P_{t+1} - P_t.
  score = TTR**gamma * LSMI   (gamma = 8, fixed). Larger = more likely manipulated. A clip is called fake if score >= tau, where
  tau comes from REAL videos only: the threshold at which at most a fraction alpha of them is flagged (see split_threshold).

This is our own implementation of the paper's equations; video/split_check.py compares it with the authors' official code
(https://github.com/mldljyh/SPLIT, MIT licence) on identical input. Preprocessing follows the paper / official code: 8 fps, a clip of
up to 2 s (16 frames, 8 if the video is short), 10 % cropped from both ends of the longer side, resize 224x224, ImageNet
normalisation. Like the official loader (cv2.imread -> albumentations) the channels are fed in BGR order; bgr=False gives RGB.

    python -m video.split score clip.mp4 [--calibration cal.json]
    python -m video.split calibrate --real-dir DIR_OF_REAL_VIDEOS --out cal.json
    python -m video.split eval --real-dir REAL --fake-dir FAKE [--real-dir2 REAL2] --out report.json
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch

GAMMA = 8.0
EPS = 1e-8
FPS = 8
CLIP_S = 2.0
SIZE = 224
CROP = 0.10
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ENCODERS = {'CLIP-16': ('clip', 'openai/clip-vit-base-patch16'), 'CLIP-32': ('clip', 'openai/clip-vit-base-patch32'),
            'XCLIP-16': ('xclip', 'microsoft/xclip-base-patch16'), 'XCLIP-32': ('xclip', 'microsoft/xclip-base-patch32'),
            'DINO-base': ('dino', 'facebook/dinov2-base'), 'DINO-large': ('dino', 'facebook/dinov2-large')}
ALPHAS = (0.001, 0.01, 0.05)


# ---------------------------------------------------------------------------------------------- the two statistics
def ttr_per_patch(tokens, eps=EPS):
    """tokens [B, T, N, D] -> TTR(n) [B, N]  (needs T >= 3)."""
    B, T, N, D = tokens.shape
    if T < 3:
        raise ValueError('TTR needs at least 3 frames')
    d1 = (tokens[:, 1:] - tokens[:, :-1]).norm(dim=-1)                     # [B, T-1, N]
    d2 = (tokens[:, 2:] - tokens[:, :-2]).norm(dim=-1)                     # [B, T-2, N]
    L1 = d1.sum(1)
    L2 = (d2.sum(1) / 2) * (T - 1) / (T - 2)
    return torch.log2(L1 + eps) - torch.log2(L2 + eps)


def lsmi(tokens, grid):
    """tokens [B, T, N, D] with N = grid[0] * grid[1] -> LSMI [B]."""
    B, T, N, D = tokens.shape
    Hp, Wp = grid
    P = tokens.reshape(B, T, Hp, Wp, D)
    M = P[:, 1:] - P[:, :-1]                                                # motion field [B, T-1, Hp, Wp, D]
    gx = (M[:, :, :, :-1] - M[:, :, :, 1:]).norm(dim=-1)
    gy = (M[:, :, :-1] - M[:, :, 1:]).norm(dim=-1)
    return (gx.mean(dim=(1, 2, 3)) + gy.mean(dim=(1, 2, 3))) / 2


def split_components(tokens, grid=None, gamma=GAMMA):
    """tokens [B, T, N, D] -> dict(ttr [B], lsmi [B], score [B], ttr_map [B, N])."""
    N = tokens.shape[2]
    grid = grid or (int(round(math.sqrt(N))),) * 2
    if grid[0] * grid[1] != N:
        raise ValueError(f'{N} tokens do not form a {grid} grid')
    tm = ttr_per_patch(tokens)
    ttr, ls = tm.mean(1), lsmi(tokens, grid)
    return dict(ttr=ttr, lsmi=ls, score=ttr.pow(gamma) * ls, ttr_map=tm)


# ---------------------------------------------------------------------------------------------- encoder + preprocessing
class SplitDetector:
    """Frozen encoder + the two statistics. Nothing is trained or fitted here (thresholds are calibrated separately)."""

    def __init__(self, encoder='XCLIP-16', device=None, gamma=GAMMA, bgr=True):
        from transformers import AutoModel, CLIPVisionModel, XCLIPVisionModel
        kind, name = ENCODERS[encoder]
        cls = {'clip': CLIPVisionModel, 'xclip': XCLIPVisionModel, 'dino': AutoModel}[kind]
        self.encoder_name, self.gamma, self.bgr = encoder, gamma, bgr
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.encoder = cls.from_pretrained(name).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def tokens(self, x):
        """x [B, T, 3, 224, 224] normalised -> patch tokens [B, T, N, D] (class token dropped)."""
        B, T = x.shape[:2]
        h = self.encoder(x.reshape(B * T, *x.shape[2:]).to(self.device)).last_hidden_state[:, 1:]
        return h.reshape(B, T, *h.shape[1:])

    @torch.no_grad()
    def components(self, x):
        return split_components(self.tokens(x), gamma=self.gamma)

    def prepare(self, frames_bgr):
        """list of HxWx3 uint8 frames (BGR as read by cv2) -> [1, T, 3, 224, 224] float tensor."""
        return prepare_frames(frames_bgr, bgr=self.bgr)

    def score_frames(self, frames_bgr):
        c = self.components(self.prepare(frames_bgr))
        return dict(ttr=float(c['ttr'][0]), lsmi=float(c['lsmi'][0]), score=float(c['score'][0]), ttr_map=c['ttr_map'][0].cpu().numpy())


def crop_longer_side(img, pct=CROP):
    """Official preprocessing: cut `pct` of the length from BOTH ends of the longer side (nothing off the shorter side)."""
    h, w = img.shape[:2]
    if w > h:
        k = int(w * pct)
        return img[:, k:w - k]
    k = int(h * pct)
    return img[k:h - k, :]


def prepare_frames(frames_bgr, bgr=True):
    mean, std = np.array(MEAN, np.float32), np.array(STD, np.float32)
    out = []
    for f in frames_bgr:
        f = crop_longer_side(f)
        f = cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        if not bgr:
            f = f[..., ::-1]
        out.append(torch.from_numpy(((f.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy()))
    return torch.stack(out).unsqueeze(0)


# ---------------------------------------------------------------------------------------------- video clips
def read_clip(path, start_s=0.0, clip_s=CLIP_S, fps=FPS):
    """Frames sampled at `fps` from start_s for clip_s seconds (BGR). Returns [] if the video cannot be read."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted({min(max(int(round((start_s + k / fps) * native)), 0), max(n - 1, 0)) for k in range(int(round(clip_s * fps)))})
    frames, i = [], 0
    while i <= (want[-1] if want else -1):
        if not cap.grab():
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
        i += 1
    cap.release()
    return frames


def clip_frame_count(frames):
    """The official loader uses 16 frames, or 8 when the clip is shorter than 16; fewer than 8 is rejected."""
    if len(frames) < 8:
        return []
    return frames[:16 if len(frames) >= 16 else 8]


def score_video(det, path, start_s=0.0, clip_s=CLIP_S):
    frames = clip_frame_count(read_clip(path, start_s, clip_s))
    return det.score_frames(frames) if frames else None


def window_scores(det, path, window_s=CLIP_S, stride_s=1.0, tau=None):
    """SPLIT score of every window of a video (for a timeline). Each record: t0, t1, ttr, lsmi, score and, when a threshold is
    given, flagged = score >= tau. The per-patch TTR map is kept as ttr_grid (rows x cols) for locating the rough region."""
    cap = cv2.VideoCapture(path)
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    out, t0 = [], 0.0
    while t0 + 1.0 <= max(dur, 1.0) and t0 < max(dur - 0.5, 0.5):
        r = score_video(det, path, t0, window_s)
        if r is not None:
            g = int(round(math.sqrt(len(r['ttr_map']))))
            rec = dict(t0=round(t0, 3), t1=round(min(t0 + window_s, dur), 3), ttr=r['ttr'], lsmi=r['lsmi'], score=r['score'],
                       ttr_grid=np.round(r['ttr_map'].reshape(g, g), 3).tolist())
            if tau is not None:
                rec['flagged'] = bool(r['score'] >= tau)
            out.append(rec)
        t0 += stride_s
    return out


# ---------------------------------------------------------------------------------------------- real-only calibration
def split_threshold(real_scores, alpha, method='paper'):
    """tau_alpha from REAL scores only.
    paper:    the smallest tau for which at most a fraction alpha of the real scores satisfy s >= tau (the paper's
              inf{tau : FPR(tau) <= alpha}); with few real videos and a tiny alpha this lies above every real score.
    official: np.quantile(real, 1 - alpha, 'higher'), what the authors' eval.py does (flags at least one real video when
              alpha * n < 1)."""
    s = np.sort(np.asarray(real_scores, float))
    n = len(s)
    if n == 0:
        raise ValueError('no real scores to calibrate on')
    if method == 'official':
        return float(np.quantile(s, 1 - alpha, method='higher'))
    k = int(math.floor(alpha * n + 1e-9))                                   # real videos we may flag
    return float(np.nextafter(s[n - 1 - k], np.inf)) if k == 0 else float(s[n - k])


def fake_recall(fake_scores, tau):
    return float((np.asarray(fake_scores, float) >= tau).mean())


def real_fpr(real_scores, tau):
    return float((np.asarray(real_scores, float) >= tau).mean())


def calibrate(real_scores, alphas=ALPHAS, method='paper'):
    return {str(a): dict(tau=split_threshold(real_scores, a, method), real_fpr=real_fpr(real_scores, split_threshold(real_scores, a, method)),
                         n_real=len(real_scores)) for a in alphas}


def harmonic_mean(xs):
    xs = [float(x) for x in xs]
    return 0.0 if (not xs or min(xs) <= 0) else len(xs) / sum(1.0 / x for x in xs)


def evaluate(real_sets, fake_scores, alphas=ALPHAS, method='paper'):
    """Service-aligned evaluation. real_sets: {name: real scores}. For every alpha: the threshold calibrated on each real set, the
    Fake Recall each gives, and their harmonic mean (the paper's cross-real threshold transfer). Also AUROC against the pooled reals."""
    from sklearn.metrics import roc_auc_score
    res = {'n_fake': len(fake_scores), 'real_sets': {k: len(v) for k, v in real_sets.items()}, 'alphas': {}}
    for a in alphas:
        per = {}
        for name, rs in real_sets.items():
            tau = split_threshold(rs, a, method)
            per[name] = dict(tau=tau, fake_recall=fake_recall(fake_scores, tau), real_fpr_on_own_set=real_fpr(rs, tau))
        res['alphas'][str(a)] = dict(per_real_set=per, harmonic_mean_fake_recall=harmonic_mean([p['fake_recall'] for p in per.values()]))
    pooled = np.concatenate([np.asarray(v, float) for v in real_sets.values()])
    res['auroc_vs_pooled_real'] = float(roc_auc_score(np.r_[np.zeros(len(pooled)), np.ones(len(fake_scores))], np.r_[pooled, fake_scores]))
    return res


# ---------------------------------------------------------------------------------------------- folder scoring + CLI
VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def score_folder(det, folder, start_s=0.0, limit=0, verbose=True):
    paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(folder) for f in fs if f.lower().endswith(VIDEO_EXT))
    paths = paths[:limit] if limit else paths
    scores, kept = [], []
    for i, p in enumerate(paths):
        r = score_video(det, p, start_s)
        if r is None:
            if verbose:
                print(f'  skipped (unreadable or < 8 frames): {p}')
            continue
        scores.append(r['score'])
        kept.append(p)
        if verbose and (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(paths)} scored')
    return np.array(scores), kept


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('score', 'calibrate', 'eval'):
        p = sub.add_parser(name)
        p.add_argument('--encoder', default='XCLIP-16', choices=list(ENCODERS))
        p.add_argument('--device', default=None)
        p.add_argument('--rgb', action='store_true', help='feed RGB instead of the official BGR order')
        if name == 'score':
            p.add_argument('video')
            p.add_argument('--calibration', default=None, help='JSON written by `calibrate`')
            p.add_argument('--alpha', default='0.01')
            p.add_argument('--windows', action='store_true', help='score every 2 s window instead of the first clip')
        else:
            p.add_argument('--real-dir', required=True)
            p.add_argument('--limit', type=int, default=0)
            p.add_argument('--out', required=True)
            p.add_argument('--method', default='paper', choices=['paper', 'official'])
        if name == 'eval':
            p.add_argument('--fake-dir', required=True)
            p.add_argument('--real-dir2', default=None, help='a second, disjoint real set for cross-real threshold transfer')
    a = ap.parse_args()
    det = SplitDetector(a.encoder, a.device, bgr=not a.rgb)
    if a.cmd == 'score':
        tau = None
        if a.calibration:
            tau = json.load(open(a.calibration))['thresholds'][a.alpha]['tau']
        if a.windows:
            for r in window_scores(det, a.video, tau=tau):
                print(f"{r['t0']:6.1f}-{r['t1']:6.1f}s  TTR {r['ttr']:.3f}  LSMI {r['lsmi']:.3f}  score {r['score']:.4g}"
                      + (f"  {'FLAGGED' if r['flagged'] else 'ok'}" if tau is not None else ''))
        else:
            r = score_video(det, a.video)
            if r is None:
                raise SystemExit('could not read at least 8 frames from the video')
            print(f"TTR {r['ttr']:.3f}  LSMI {r['lsmi']:.3f}  SPLIT score {r['score']:.4g}"
                  + (f"  -> {'MANIPULATED' if r['score'] >= tau else 'real'} (tau {tau:.4g} at alpha {a.alpha})" if tau is not None else '  (no calibration: pass --calibration)'))
    elif a.cmd == 'calibrate':
        s, kept = score_folder(det, a.real_dir, limit=a.limit)
        json.dump(dict(encoder=a.encoder, method=a.method, n_real=len(s), real_dir=a.real_dir, thresholds=calibrate(s, ALPHAS, a.method),
                       real_scores=s.tolist()), open(a.out, 'w'), indent=1)
        print(f'calibrated on {len(s)} real videos -> {a.out}')
    else:
        rs, _ = score_folder(det, a.real_dir, limit=a.limit)
        fs, _ = score_folder(det, a.fake_dir, limit=a.limit)
        sets = {os.path.basename(a.real_dir.rstrip('/')): rs}
        if a.real_dir2:
            sets[os.path.basename(a.real_dir2.rstrip('/')) + '_2'] = score_folder(det, a.real_dir2, limit=a.limit)[0]
        res = evaluate(sets, fs, ALPHAS, a.method)
        res.update(encoder=a.encoder, method=a.method, fake_dir=a.fake_dir)
        json.dump(res, open(a.out, 'w'), indent=1)
        for al, v in res['alphas'].items():
            print(f"FPR {float(al):.1%}: Fake Recall (harmonic mean over real sets) {v['harmonic_mean_fake_recall']:.3f}")
        print(f"AUROC {res['auroc_vs_pooled_real']:.3f}  -> {a.out}")


if __name__ == '__main__':
    main()
