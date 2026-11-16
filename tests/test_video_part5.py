"""Part 5 checks: metrics, manifest / folder reading, and a full evaluation run on cached synthetic records
(no model, no real video). Run: python tests/test_video_part5.py"""
import csv
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.eval import auc, cache_path, evaluate_video, interval_iou, read_manifest, scan_root, summarize

TYPES = ['real', 'face_swap', 'splice', 'diffusion_local', 'img2img', 'generated']


def rec(t, p):
    types = {k: 0.0 for k in TYPES}
    types['face_swap' if p >= .5 else 'real'] = 1.0
    return dict(t=t, frame=int(t * 25), scene=0, cut=False, track=0, box=None, p_fake=p, type='face_swap' if p >= .5 else 'real',
                types=types, filters={'blur': .1, 'jpeg': .2}, forgery_area=.2 if p >= .5 else 0.0)


def write_cache(path, ps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(json.dumps(dict(video='x', fps=25, duration=float(len(ps)))) + '\n')
        f.writelines(json.dumps(rec(float(i), p)) + '\n' for i, p in enumerate(ps))


def main():
    # 1. AUC: perfect, inverted, tied, one class missing
    assert auc([0, 0, 1, 1], [.1, .2, .8, .9]) == 1.0
    assert auc([0, 0, 1, 1], [.9, .8, .2, .1]) == 0.0
    assert auc([0, 1, 0, 1], [.5, .5, .5, .5]) == 0.5
    assert auc([1, 1], [.2, .3]) is None
    assert abs(auc([0, 0, 1, 1], [.1, .6, .5, .9]) - 0.75) < 1e-9

    # 2. interval IoU: overlap, disjoint, merged inputs
    assert abs(interval_iou([(2, 6)], [(4, 8)]) - 2 / 6) < 1e-9
    assert interval_iou([(0, 1)], [(2, 3)]) == 0.0
    assert interval_iou([(0, 2), (1, 3)], [(0, 3)]) == 1.0

    with tempfile.TemporaryDirectory() as d:
        # 3. folder scan: FF++-style layout, real folders vs method folders
        for sub, n in (('original', 2), ('Deepfakes', 2), ('Face2Face', 1)):
            os.makedirs(os.path.join(d, 'vids', sub))
            for i in range(n):
                open(os.path.join(d, 'vids', sub, f'{i}.mp4'), 'w').close()
        items = scan_root(os.path.join(d, 'vids'))
        assert [i['label'] for i in items].count(0) == 2 and [i['label'] for i in items].count(1) == 3
        assert {i['method'] for i in items if i['label'] == 1} == {'Deepfakes', 'Face2Face'}

        # 4. synthetic study: 3 real (clean), 2 fake (obvious), 1 localized fake inside a clean video
        cache = os.path.join(d, 'cache')
        spec = [('real', 0, [.05, .1, .04, .08, .06, .03, .07, .05]), ('real', 0, [.1, .05, .08, .12, .06, .09, .04, .07]),
                ('real', 0, [.03] * 8),
                ('Deepfakes', 1, [.95, .9, .97, .93, .96, .92, .94, .9]), ('Deepfakes', 1, [.85, .9, .8, .92, .88, .9, .86, .91]),
                ('Local', 1, [.05] * 8 + [.93, .95, .9, .96] + [.04] * 8)]
        man = os.path.join(d, 'videos.csv')
        with open(man, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video', 'label', 'method', 'fake_start', 'fake_end'])
            for k, (m, lab, ps) in enumerate(spec):
                vid = os.path.join(d, 'v', f'clip{k}.mp4')
                os.makedirs(os.path.dirname(vid), exist_ok=True)
                write_cache(cache_path(cache, vid), ps)
                w.writerow([vid, lab, m, 8 if m == 'Local' else '', 11 if m == 'Local' else ''])
        items = read_manifest(man)
        assert len(items) == 6 and items[5]['fake_start'] == 8.0

        out = os.path.join(d, 'out')
        r = subprocess.run([sys.executable, '-m', 'video.eval', '--manifest', man, '--out-dir', out, '--cache-dir', cache],
                           capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert r.returncode == 0, r.stdout + r.stderr
        rep = json.load(open(os.path.join(out, 'report.json')))
        assert rep['n_videos'] == 6 and rep['n_real'] == 3 and rep['n_fake'] == 3 and rep['skipped'] == 0
        assert rep['auc']['segment'] == 1.0, rep['auc']
        assert rep['overall']['tpr'] == 1.0 and rep['overall']['fpr'] == 0.0 and rep['overall']['accuracy'] == 1.0, rep['overall']
        # the localized fake is averaged away by the mean score but found by the segment score
        rows = list(csv.DictReader(open(os.path.join(out, 'results.csv'))))
        loc = [x for x in rows if x['method'] == 'Local'][0]
        assert float(loc['score_mean']) < 0.5 <= float(loc['score_segment']) and loc['verdict'] == 'fake', loc
        assert 0.5 < float(loc['iou']) <= 1.0, loc
        assert rep['localisation']['n'] == 1 and set(rep['by_method']) == {'Deepfakes', 'Local'}
        assert os.path.exists(os.path.join(out, 'report.md'))

        # 5. a video with no cache and no model is reported as skipped, not silently dropped, and does not stop the run
        with open(man, 'a', newline='') as f:
            csv.writer(f).writerow([os.path.join(d, 'v', 'missing.mp4'), 1, 'Deepfakes', '', ''])
        r = subprocess.run([sys.executable, '-m', 'video.eval', '--manifest', man, '--out-dir', out, '--cache-dir', cache],
                           capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert r.returncode == 0 and 'SKIPPED' in r.stdout, r.stdout + r.stderr
        assert json.load(open(os.path.join(out, 'report.json')))['skipped'] == 1

    print('part 5: all checks passed')


if __name__ == '__main__':
    main()
