"""Part 4 checks: the video-level explanation. A stub stands in for the LLM (no model, no GPU).
Run: python tests/test_video_part4.py   (needs the face image used by test_video_part1)"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_video_part1 import make_video          # noqa: E402
from test_video_part3 import rec                 # noqa: E402
from video.explain import explain_video, llm_verdict, segment_hint   # noqa: E402
from video.timeline import build_timeline        # noqa: E402

FAKE_TXT = 'This image is fake. The face has been replaced with another identity (a face-swap). The manipulated region covers the eyes and mouth.'
REAL_TXT = 'This image is real. No manipulated region was found.'


def setup(rows, header=None):
    d = tempfile.mkdtemp()
    v = os.path.join(d, 'clip.mp4')
    make_video(v)
    header = header or dict(video=v, fps=25.0, duration=8.0, cuts=[0, 100])
    tl = build_timeline(header, rows)
    fj, tj = os.path.join(d, 'clip.frames.jsonl'), os.path.join(d, 'clip.timeline.json')
    with open(fj, 'w') as f:
        f.write(json.dumps(header) + '\n' + '\n'.join(json.dumps(r) for r in rows) + '\n')
    json.dump(tl, open(tj, 'w'))
    return d, v, fj, tj, tl


def main():
    clean_rows = [rec(t, 0.05, 0, 0) for t in range(4)]
    fake_rows = clean_rows + [rec(t, p, 1, 1) for t, p in zip(range(4, 8), (0.92, 0.95, 0.9, 0.94))]
    d, v, fj, tj, tl = setup(fake_rows)

    # 1. a fake shot: the text states verdict, time range, type, and carries the LLM's key-frame explanation; the LLM got a face crop + hint
    seen = []

    def ask_fake(img, hint):
        seen.append((img.size, hint))
        return FAKE_TXT
    r = explain_video(v, fj, tj, ask_fake)
    assert r['verdict'] == 'fake' and len(r['segments']) == 1 and not r['disagreements'], r
    t = r['text']
    assert 'judged fake' in t and '4.0 s to 7.0 s' in t and 'a face swap' in t and 'face 1' in t and FAKE_TXT in t, t
    seg = r['segments'][0]
    assert seg['key_frame']['frame'] == 125 and seg['agrees'] is True and seg['llm_verdict'] == 'fake', seg      # the peak frame (t = 5 s)
    (w, h), hint = seen[0]
    assert 180 <= w <= 340 and 180 <= h <= 340 and 'mean fake probability' in hint and '4.0 s to 7.0 s' in hint, (seen[0])
    assert len(seen) == 1 and json.loads(json.dumps(r))['text'] == t

    # 2. the LLM disagrees with the timeline: reported, the timeline verdict stands
    r = explain_video(v, fj, tj, lambda img, hint: REAL_TXT)
    assert r['verdict'] == 'fake' and len(r['disagreements']) == 1 and r['segments'][0]['agrees'] is False
    assert 'Disagreement' in r['text'] and 'timeline verdict stands' in r['text'], r['text']

    # 3. no LLM: the rule-based part alone is complete
    r = explain_video(v, fj, tj, None)
    assert not r['llm_used'] and r['segments'][0]['frame_explanation'] is None and '4.0 s to 7.0 s' in r['text'] and 'image explainer' not in r['text']

    # 4. a clean video: no segment, the cleanest frame is shown to the LLM, no false alarm
    d2, v2, fj2, tj2, tl2 = setup(clean_rows + [rec(t, 0.04, 1, 1) for t in range(4, 8)])
    calls = []
    r = explain_video(v2, fj2, tj2, lambda img, hint: calls.append(hint) or REAL_TXT)
    assert r['verdict'] == 'real' and not r['segments'] and 'No manipulation was found' in r['text'] and len(calls) == 1
    assert r['clean_frame']['agrees'] is True and 'Cleanest frame' in r['text']
    r = explain_video(v2, fj2, tj2, lambda img, hint: FAKE_TXT)           # LLM says fake on a clean video: reported
    assert r['verdict'] == 'real' and any('judges the cleanest frame' in x for x in r['disagreements'])

    # 5. two faces: only face 1 is described; an isolated spike is noted and not counted
    rows2 = [rec(t, 0.04, 0, 0) for t in range(8)] + [rec(t, 0.95, 1, 1) for t in range(8)]
    rows2 = [dict(x, track=0 if i < 8 else 1, scene=0) for i, x in enumerate(rows2)]
    d3, v3, fj3, tj3, _ = setup(rows2)
    r = explain_video(v3, fj3, tj3, ask_fake)
    assert len(r['segments']) == 1 and r['segments'][0]['face'] == 'face 1' and 'face 0' not in r['text'], r['text']
    d4, v4, fj4, tj4, tl4 = setup([rec(t, 0.05, 0, 0) for t in range(3)] + [rec(3, 0.97, 0, 0)] + [rec(t, 0.05, 0, 0) for t in range(4, 8)])
    r = explain_video(v4, fj4, tj4, None)
    assert tl4['isolated_spikes'] and r['notes'] and 'did not form a segment' in r['text']

    # 6. helpers
    assert llm_verdict('This image is fake.') == 'fake' and llm_verdict('This image is real.') == 'real'
    assert llm_verdict('The picture has not been tampered with') == 'real' and llm_verdict('Content has been pasted in from another image') == 'fake'
    assert llm_verdict('') is None and llm_verdict(None) is None
    assert 'track' not in segment_hint(dict(track='track 1', start=1.0, end=2.0, n_frames=3, mean_p=0.9, type='splice')) and 'splicing' in segment_hint(
        dict(track='track 1', start=1.0, end=2.0, n_frames=3, mean_p=0.9, type='splice'))

    # 7. the CLI with --no-llm writes the JSON and the text
    out = os.path.join(d, 'clip.explanation.json')
    p = subprocess.run([sys.executable, '-m', 'video.explain', v, fj, tj, '--out', out, '--no-llm'], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert p.returncode == 0 and os.path.exists(out) and os.path.exists(out[:-5] + '.txt'), p.stderr[-400:]
    assert json.load(open(out))['verdict'] == 'fake' and 'judged fake' in p.stdout
    print(f'OK  8 scenarios; example text:\n{explain_video(v, fj, tj, ask_fake)["text"]}')


if __name__ == '__main__':
    main()
