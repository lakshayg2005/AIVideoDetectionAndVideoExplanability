"""SPLIT unit checks (formulas, preprocessing, real-only calibration). No network, no model download.
Run: python tests/test_split.py"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.split import (clip_frame_count, crop_longer_side, evaluate, fake_recall, harmonic_mean, lsmi, prepare_frames,  # noqa: E402
                         real_fpr, split_components, split_threshold, ttr_per_patch)


def naive_ttr(p, eps=1e-8):
    """p [T, D] one patch trajectory, straight from the paper's equations (2)-(4)."""
    T = len(p)
    L1 = sum(np.linalg.norm(p[t + 1] - p[t]) for t in range(T - 1))
    L2h = sum(np.linalg.norm(p[t + 2] - p[t]) for t in range(T - 2))
    L2 = (L2h / 2) * (T - 1) / (T - 2)
    return math.log2(L1 + eps) - math.log2(L2 + eps)


def naive_lsmi(P):
    """P [T, Hp, Wp, D] - equations (6)-(9) with explicit loops."""
    T, Hp, Wp, D = P.shape
    gx, gy = [], []
    for t in range(T - 1):
        M = P[t + 1] - P[t]
        for i in range(Hp):
            for j in range(Wp - 1):
                gx.append(np.linalg.norm(M[i, j] - M[i, j + 1]))
        for i in range(Hp - 1):
            for j in range(Wp):
                gy.append(np.linalg.norm(M[i, j] - M[i + 1, j]))
    return 0.5 * (np.mean(gx) + np.mean(gy))


def main():
    rng = np.random.default_rng(0)
    T, Hp, Wp, D = 8, 4, 5, 6
    P = rng.normal(size=(T, Hp, Wp, D)).astype(np.float64)
    tok = torch.from_numpy(P.reshape(1, T, Hp * Wp, D))

    # 1. vectorised TTR and LSMI equal the equations written out with loops
    tm = ttr_per_patch(tok)[0].numpy()
    ref = np.array([naive_ttr(P.reshape(T, Hp * Wp, D)[:, n]) for n in range(Hp * Wp)])
    assert np.allclose(tm, ref, atol=1e-9), np.abs(tm - ref).max()
    assert abs(float(lsmi(tok, (Hp, Wp))[0]) - naive_lsmi(P)) < 1e-9
    c = split_components(tok, (Hp, Wp))
    assert abs(float(c['score'][0]) - float(c['ttr'][0]) ** 8 * float(c['lsmi'][0])) < 1e-9

    # 2. smooth constant-velocity motion: L1 == L2, so TTR = 0; a zigzag trajectory has TTR > 0; uniform motion has LSMI = 0
    v = rng.normal(size=(1, 1, Hp * Wp, D))
    lin = torch.from_numpy(np.arange(T).reshape(1, T, 1, 1) * v + 0.5)
    assert float(ttr_per_patch(lin).abs().max()) < 1e-6
    uni = torch.from_numpy(np.arange(T).reshape(1, T, 1, 1) * rng.normal(size=(1, 1, 1, D)) + np.zeros((1, 1, Hp * Wp, 1)))
    assert float(lsmi(uni, (Hp, Wp))[0]) < 1e-9, 'the same motion in every patch is spatially coherent: LSMI = 0'
    assert float(lsmi(lin, (Hp, Wp))[0]) > 0.1, 'a different motion in every patch is incoherent'

    zig = torch.from_numpy(((np.arange(T) % 2).reshape(1, T, 1, 1) * v * 3.0) * np.ones((1, 1, Hp * Wp, 1)))
    assert float(ttr_per_patch(zig).mean()) > 0.5, float(ttr_per_patch(zig).mean())
    # patch-level: flicker in ONE patch raises the mean TTR and LSMI, i.e. a local edit is visible although most of the frame is smooth
    loc = lin.clone()
    loc[:, 1::2, 7] += 4.0
    assert float(ttr_per_patch(loc)[0, 7]) > 1.0 and float(ttr_per_patch(loc).mean()) > float(ttr_per_patch(lin).mean()) + 0.04
    assert float(lsmi(loc, (Hp, Wp))[0]) > float(lsmi(lin, (Hp, Wp))[0])
    loc_u = uni.clone()
    loc_u[:, 1::2, 7] += 4.0                                                    # an edit in one patch of an otherwise coherent scene
    assert float(lsmi(loc_u, (Hp, Wp))[0]) > 0.5
    try:
        ttr_per_patch(tok[:, :2])
        raise AssertionError('T < 3 must be rejected')
    except ValueError:
        pass

    # 3. preprocessing: the longer side loses 10 % at each end; portrait crops the height; BGR vs RGB flips channels only
    land = np.zeros((100, 200, 3), np.uint8)
    port = np.zeros((200, 100, 3), np.uint8)
    assert crop_longer_side(land).shape == (100, 160, 3) and crop_longer_side(port).shape == (160, 100, 3)
    img = np.zeros((120, 120, 3), np.uint8)
    img[..., 0] = 255                                                          # pure blue in BGR
    xb, xr = prepare_frames([img] * 3, bgr=True), prepare_frames([img] * 3, bgr=False)
    assert xb.shape == (1, 3, 3, 224, 224)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    assert abs(float(xb[0, 0, 0, 5, 5]) - (1 - mean[0]) / std[0]) < 1e-4 and abs(float(xr[0, 0, 2, 5, 5]) - (1 - mean[2]) / std[2]) < 1e-4
    assert len(clip_frame_count(list(range(20)))) == 16 and len(clip_frame_count(list(range(10)))) == 8 and clip_frame_count(list(range(7))) == []

    # 4. real-only calibration: the threshold flags at most alpha of the REAL scores
    real = rng.lognormal(0.0, 1.0, 1000)
    fake = rng.lognormal(2.5, 1.0, 400)
    for a in (0.001, 0.01, 0.05):
        t = split_threshold(real, a)
        assert real_fpr(real, t) <= a + 1e-12, (a, real_fpr(real, t))
        assert real_fpr(real, t) >= a - 1.5 / len(real)                          # and it is not needlessly strict
    small = rng.lognormal(0.0, 1.0, 100)
    assert real_fpr(small, split_threshold(small, 0.001)) == 0.0                 # alpha * n < 1: flag nothing real, so recall is paid
    assert real_fpr(small, split_threshold(small, 0.001, 'official')) == 0.01   # the official quantile always flags the top real score
    assert real_fpr(small, split_threshold(small, 0.05)) == 0.05
    # 5. evaluation: harmonic mean over two real sets, and AUROC
    ev = evaluate({'A': real[:500], 'B': real[500:]}, fake)
    h = ev['alphas']['0.01']
    assert abs(h['harmonic_mean_fake_recall'] - harmonic_mean([p['fake_recall'] for p in h['per_real_set'].values()])) < 1e-12
    assert 0.9 < ev['auroc_vs_pooled_real'] <= 1.0 and fake_recall(fake, 0.0) == 1.0 and harmonic_mean([0.5, 0.0]) == 0.0
    print('OK  formulas, preprocessing, real-only calibration, evaluation')


if __name__ == '__main__':
    main()
