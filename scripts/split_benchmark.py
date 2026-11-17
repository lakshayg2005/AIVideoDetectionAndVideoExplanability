"""Benchmark SPLIT on a real / fake video collection, with real-only calibration.

    python scripts/split_benchmark.py --real-a DIR --real-b DIR --fake-root DIR --encoder XCLIP-16 --out report.json

real-a and real-b are two disjoint folders of real videos: the threshold is calibrated on real-a (at 1 / 5 / 10 / 20 % real
false-positive rate) and checked on real-b; fakes never touch the calibration. fake-root holds one sub-folder per generator /
manipulation method, which gives the per-method table. Reports fake recall at each rate, AUROC against real-b and the
per-method recall and AUROC.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video.split as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-a', required=True)
    ap.add_argument('--real-b', required=True)
    ap.add_argument('--fake-root', required=True)
    ap.add_argument('--encoder', default='XCLIP-16', choices=list(S.ENCODERS))
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    det = S.SplitDetector(a.encoder, a.device, bgr=True)
    ra, _ = S.score_folder(det, a.real_a, verbose=False)
    rb, _ = S.score_folder(det, a.real_b, verbose=False)
    fs, kept = S.score_folder(det, a.fake_root, verbose=False)
    gen = np.array([os.path.basename(os.path.dirname(p)) for p in kept])
    out = {'encoder': a.encoder, 'n_real_a': len(ra), 'n_real_b': len(rb), 'n_fake': len(fs), 'by_alpha': {}, 'by_method': {}}
    for al in (0.01, 0.05, 0.10, 0.20):
        ta = S.split_threshold(ra, al)
        out['by_alpha'][str(al)] = {'fake_recall': float((fs >= ta).mean()), 'real_b_fpr': float((rb >= ta).mean())}
    lab = lambda n_f: np.r_[np.zeros(len(rb)), np.ones(n_f)]
    out['auroc_vs_real_b'] = float(roc_auc_score(lab(len(fs)), np.r_[rb, fs]))
    ta = S.split_threshold(ra, 0.05)
    for g in sorted(set(gen)):
        m = gen == g
        out['by_method'][g] = {'n': int(m.sum()), 'recall_at_5pct': float((fs[m] >= ta).mean()),
                               'auroc_vs_real_b': float(roc_auc_score(lab(int(m.sum())), np.r_[rb, fs[m]]))}
    out['median_score'] = {'real': float(np.median(np.r_[ra, rb])), 'fake': float(np.median(fs))}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
