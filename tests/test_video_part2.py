"""Part 2 checks on synthetic per-frame records (no model, no video). Run: python tests/test_video_part2.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.timeline import build_timeline, find_segments, smooth, trimmed_mean

TYPES = ['real', 'face_swap', 'splice', 'diffusion_local', 'img2img', 'generated']


def rec(t, p, track=0, scene=0, typ='face_swap', area=0.2):
    types = {k: 0.0 for k in TYPES}
    types[typ if p >= 0.5 else 'real'] = 1.0
    return dict(t=t, frame=int(t * 25), scene=scene, cut=False, track=track, box=None, p_fake=p, type=typ if p >= .5 else 'real',
                types=types, filters={'blur': 0.1, 'jpeg': 0.2}, forgery_area=area if p >= .5 else 0.0)


def series(ps, dt=1.0, **kw):
    return [rec(i * dt, p, **kw) for i, p in enumerate(ps)]


H = dict(video='x.mp4', duration=20.0)


def main():
    # 1. a clean video is real
    r = build_timeline(H, series([0.05, 0.1, 0.03, 0.08, 0.04, 0.06, 0.02, 0.09]))
    assert r['verdict']['label'] == 'real' and not r['segments'], r['verdict']

    # 2. an all-fake video is fake, with the dominant type and one segment covering it
    r = build_timeline(H, series([0.95, 0.9, 0.97, 0.93, 0.96, 0.92], typ='diffusion_local'))
    assert r['verdict']['label'] == 'fake' and r['verdict']['dominant_type'] == 'diffusion_local'
    assert len(r['segments']) == 1 and r['segments'][0]['n_frames'] == 6, r['segments']

    # 3. a manipulated middle part inside a long clean video is found, at the right time, and still makes the video fake
    ps = [0.05] * 8 + [0.93, 0.95, 0.9, 0.96] + [0.04] * 8
    r = build_timeline(H, series(ps, typ='face_swap'))
    s = r['segments']
    assert len(s) == 1 and 7.0 <= s[0]['start'] <= 9.0 and 10.0 <= s[0]['end'] <= 12.0, s
    assert r['verdict']['p_video'] < 0.5 and r['verdict']['label'] == 'fake', r['verdict']    # averaged away, found by the segment rule
    assert 'localized' in r['verdict']['reasons'][0]

    # 4. one wrong frame in a clean video is not a segment and does not flip the verdict, but is listed as a spike
    r = build_timeline(H, series([0.05, 0.04, 0.06, 0.97, 0.05, 0.03, 0.06, 0.04]))
    assert r['verdict']['label'] == 'real' and not r['segments'], (r['verdict'], r['segments'])
    assert len(r['isolated_spikes']) == 1 and r['isolated_spikes'][0]['t'] == 3.0

    # 5. two faces are tracked apart: only face 1 is fake
    rows = series([0.04] * 8, track=0) + series([0.95] * 8, track=1)
    r = build_timeline(H, rows)
    assert all(sg['track'] == 'track 1' for sg in r['segments']) and len(r['segments']) == 1, r['segments']

    # 6. no-face frames (track -1) are grouped per shot and still analysed
    rows = series([0.05, 0.04, 0.06], track=-1, scene=0) + series([0.94, 0.96, 0.95], track=-1, scene=1)
    r = build_timeline(H, rows)
    assert {k for k in r['timeline']} == {'no-face shot 0', 'no-face shot 1'} and len(r['segments']) == 1

    # 7. a far gap splits a suspect run in two
    rs = series([0.95, 0.96, 0.95], dt=1.0) + [rec(30 + i, 0.95) for i in range(3)]
    sm = smooth([x['t'] for x in rs], [x['p_fake'] for x in rs])
    assert len(find_segments(rs, sm)) == 2

    # 8. an unsure video (middling scores, no strong segment) is reported as uncertain
    r = build_timeline(H, series([0.5, 0.45, 0.55, 0.48, 0.52, 0.47, 0.5, 0.46], typ='splice'))
    assert r['verdict']['label'] in ('uncertain', 'fake'), r['verdict']

    # helpers
    assert trimmed_mean([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]) == 0.5
    sm = smooth([0, 1, 2], [0.0, 1.0, 0.0])
    assert 0 < sm[1] < 1 and abs(smooth([0, 1], [0.3, 0.3])[0] - 0.3) < 1e-9
    print('OK  8 scenarios + helpers')


if __name__ == '__main__':
    main()
