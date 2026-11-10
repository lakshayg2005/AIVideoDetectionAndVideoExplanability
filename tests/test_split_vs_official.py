"""Our SPLIT against the authors' official SPLIT_model (https://github.com/mldljyh/SPLIT) on identical input and weights.
Needs the cloned repo (default /home/Aditya/bp/SPLIT/SPLIT, or set SPLIT_OFFICIAL) and the XCLIP weights from the Hugging Face hub.
Run: python tests/test_split_vs_official.py [encoder]      (CPU is fine)"""
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.split import SplitDetector, split_threshold  # noqa: E402

OFFICIAL = os.environ.get('SPLIT_OFFICIAL', '/home/Aditya/bp/SPLIT/SPLIT')


def load_official():
    spec = importlib.util.spec_from_file_location('official_split_model', os.path.join(OFFICIAL, 'models', 'SPLIT_model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SPLIT_model


def official_fake_recall_at_fpr(real_scores, fake_scores, target_fpr):      # eval.py, evaluate_split(), verbatim logic
    threshold = np.quantile(real_scores, 1 - target_fpr, method='higher')
    return float((fake_scores >= threshold).mean())


def main():
    enc = sys.argv[1] if len(sys.argv) > 1 else 'XCLIP-16'
    if not os.path.isdir(OFFICIAL):
        print(f'SKIP  official repo not found at {OFFICIAL}')
        return
    torch.manual_seed(0)
    off = load_official()(encoder_type=enc).eval()
    ours = SplitDetector(enc, device='cpu')
    worst = 0.0
    for T in (8, 16):
        x = torch.randn(2, T, 3, 224, 224)
        with torch.no_grad():
            _, ttr_o, lsmi_o = off._compute_components(x, return_global=False)
            s_o = off.forward_score(x)
        c = ours.components(x)
        for name, a, b in (('TTR', ttr_o.mean(1), c['ttr']), ('LSMI', lsmi_o, c['lsmi']), ('score', s_o, c['score'])):
            rel = float(((a - b).abs() / (a.abs() + 1e-12)).max())
            worst = max(worst, rel)
            print(f'  T={T:2d} {name:5s} official {a.tolist()}  ours {b.tolist()}  max rel diff {rel:.2e}')
            assert rel < 1e-3, (T, name, rel)
    # the calibration rule the official eval.py uses is our 'official' method
    rng = np.random.default_rng(1)
    real, fake = rng.lognormal(0, 1, 800), rng.lognormal(2, 1, 300)
    for a in (0.001, 0.01, 0.05):
        t = split_threshold(real, a, 'official')
        assert float((fake >= t).mean()) == official_fake_recall_at_fpr(real, fake, a)
    print(f'OK  {enc}: TTR, LSMI and score agree with the official implementation (worst relative difference {worst:.1e}); '
          f'the official thresholding rule is reproduced')


if __name__ == '__main__':
    main()
