"""Part 6 checks: the UI's video pipeline function end to end on a synthetic 8 s two-shot video, with a stub engine in place of the
forensic model and no language model (the Part 4 stub path is tested in test_video_part4). Also builds the Gradio page.
Run: python tests/test_video_part6.py   (needs the face image used by test_video_part1)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_video_part1 import make_video          # noqa: E402
import app.forensic_webui as ui                  # noqa: E402

TYPES = ['real', 'face_swap', 'splice', 'diffusion_local', 'img2img', 'generated']


class StubForensic:
    """Clean for the first calls, fake afterwards (frames are analysed in time order)."""
    def __init__(self, n_clean):
        self.n, self.calls = n_clean, 0

    def analyze(self, img):
        self.calls += 1
        fake = self.calls > self.n
        types = {k: 0.0 for k in TYPES}
        types['face_swap' if fake else 'real'] = 1.0
        m = np.zeros((img.size[1], img.size[0]), np.float32)
        if fake:
            m[m.shape[0] // 4:m.shape[0] // 2, m.shape[1] // 4:m.shape[1] // 2] = 0.9
        return {'p_fake': 0.93 if fake else 0.05, 'types': types, 'filters': {'blur': 0.1, 'jpeg': 0.2}, 'forgery_map': m, 'filter_map': m * 0}


def main():
    d = tempfile.mkdtemp()
    v = os.path.join(d, 'clip.mp4')
    make_video(v)
    ui.LLM, ui.SELF_EXPLAIN = None, False
    ui.FORENSIC = StubForensic(n_clean=8)
    verdict, plot, ann, keys, segs, text, meta = ui.analyze_video(v, 16, True, lambda *a, **k: None)
    assert 'LIKELY FAKE' in verdict and 'face-swap' in verdict.lower() or 'FAKE' in verdict, verdict
    assert os.path.getsize(plot) > 1000 and plot.endswith('.png')
    assert ann and os.path.getsize(ann) > 10000 and ann.endswith('.mp4'), ann
    assert keys and os.path.getsize(keys) > 1000
    assert len(segs) >= 1 and len(segs[0]) == 6 and float(segs[0][1].split('-')[0]) >= 3.0, segs    # the segment starts in the second shot
    assert 'off' in text and 'segment' in text.lower(), text
    assert 'sampled' in meta or 'records' in meta, meta
    # a clean video is real and has no segment
    ui.FORENSIC = StubForensic(n_clean=10 ** 6)
    verdict, _, _, _, segs, text, _ = ui.analyze_video(v, 16, True, lambda *a, **k: None)
    assert 'LIKELY REAL' in verdict and segs == [], (verdict, segs)
    # the page builds
    ui.build_ui()
    print('part 6: all checks passed')


if __name__ == '__main__':
    main()
