"""Part 4: one explanation for a whole video.

    python -m video.explain clip.mp4 clip.frames.jsonl clip.timeline.json --out clip.explanation.json [--no-llm]
        [--model-path <LoRA dir> --model-base <forensic base> --fp16 --hint-style v3]

The verdict and *where / when* come from the timeline (Part 2), so the text cannot contradict the numbers: a rule-based summary
states the verdict, why, and every suspect segment (face, time range, edit type, strength). For each of the strongest segments the
image explainer (Stage C LLM) is then asked about that segment's most telling frame, with the segment's evidence written in
front of its question as a hint, exactly the way the image UI gives it the heatmap hint. If the LLM's own verdict on that frame
disagrees with the timeline, the disagreement is reported instead of hidden. With --no-llm (or no GPU) only the rule-based part
is produced. The LLM is passed in as a function `ask(image, hint) -> text`, so everything here is testable without a model.
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image

from video.faces import crop_box
from video.render import read_frames, track_name
from video.timeline import load_records

TYPE_TEXT = {'face_swap': 'a face swap / reenactment', 'splice': 'splicing / copy-move', 'diffusion_local': 'diffusion inpainting (a local edit)',
             'img2img': 'diffusion re-rendering (img2img)', 'generated': 'a fully AI-generated image', 'real': 'no manipulation'}


def fmt_range(a, b):
    return f'{a:.1f} s' if abs(b - a) < 1e-6 else f'{a:.1f} s to {b:.1f} s'


def type_text(t):
    return TYPE_TEXT.get(t, t or 'an unspecified manipulation')


def face_name(track):
    """'track 3' -> 'face 3'; 'no-face shot 1' -> 'shot 1 (no face found)'."""
    m = re.match(r'track (\d+)', track)
    if m:
        return f'face {m.group(1)}'
    m = re.match(r'no-face shot (\d+)', track)
    return f'shot {m.group(1)} (no face found, whole frame)' if m else track


def segment_hint(seg):
    """The segment's evidence in words: written in front of the LLM's question about the segment's key frame."""
    return (f"Video evidence: this {'face' if seg['track'].startswith('track') else 'frame'} is flagged from {fmt_range(seg['start'], seg['end'])} "
            f"({seg['n_frames']} sampled frames, mean fake probability {seg['mean_p']:.2f}); the type head says {type_text(seg['type'])}.")


def llm_verdict(text):
    """'fake' / 'real' / None from an explainer answer (the image explainer's two answer styles)."""
    t = (text or '').lower()
    if 'has not been tampered' in t or 'taken directly by the camera' in t:
        return 'real'
    if 'has been tampered' in t or 'pasted in' in t:
        return 'fake'
    m = re.search(r'\b(fake|real)\b', t)
    return m.group(1) if m else None


def peak_record(seg, rows):
    by = {(track_name(r), r['frame']): r for r in rows}
    cand = [by[(seg['track'], f)] for f in seg['frames'] if (seg['track'], f) in by]
    return max(cand, key=lambda r: r['p_fake']) if cand else None


def crop_of(frame, rec):
    img = Image.fromarray(frame)
    return img.crop(crop_box(rec['box'], img.size)) if rec['box'] is not None else img


def summary_text(tl):
    v, segs = tl['verdict'], tl['segments']
    p = v['p_video']
    if v['label'] == 'fake':
        head = f"This video is judged fake (video score {p:.2f}): {v['reasons'][0]}."
    elif v['label'] == 'uncertain':
        head = f"The evidence on this video is inconclusive (video score {p:.2f}): {v['reasons'][0]}."
    else:
        head = f"No manipulation was found in this video (video score {p:.2f}): {v['reasons'][0]}."
    if not segs:
        return head
    lines = [f"The evidence is concentrated in {len(segs)} suspect segment{'s' if len(segs) > 1 else ''}:"]
    for s in segs:
        lines.append(f"- {face_name(s['track'])}, {fmt_range(s['start'], s['end'])}: {type_text(s['type'])}, mean fake probability {s['mean_p']:.2f} "
                     f"over {s['n_frames']} sampled frames, the heatmap marks about {100 * s['forgery_area']:.0f}% of the face.")
    return head + ' ' + lines[0] + '\n' + '\n'.join(lines[1:])


def explain_video(video, frames_jsonl, timeline_json, ask=None, n_key=3):
    """Returns a JSON-serialisable dict: summary, per-segment records (with the LLM's frame explanation when `ask` is given),
    disagreements between the LLM and the timeline, notes and the final text."""
    header, rows = load_records(frames_jsonl)
    tl = json.load(open(timeline_json))
    segs = sorted(tl['segments'], key=lambda s: -s['mean_p'])
    chosen = [(s, peak_record(s, rows)) for s in segs[:n_key]]
    clean = None
    if not segs:
        clean = min(rows, key=lambda r: r['p_fake'])
        chosen = []
    frames = read_frames(video, [r['frame'] for _, r in chosen if r] + ([clean['frame']] if clean else []))

    out_segments, disagreements = [], []
    for s in segs:
        rec = dict(face=face_name(s['track']), start=s['start'], end=s['end'], type=s['type'], mean_p=s['mean_p'], n_frames=s['n_frames'],
                   key_frame=None, frame_explanation=None, llm_verdict=None, agrees=None)
        out_segments.append(rec)
    for s, r in chosen:
        rec = out_segments[segs.index(s)]
        if r is None or r['frame'] not in frames:
            continue
        rec['key_frame'] = dict(frame=r['frame'], t=r['t'], p_fake=r['p_fake'])
        if ask is not None:
            txt = ask(crop_of(frames[r['frame']], r), segment_hint(s))
            rec.update(frame_explanation=txt, llm_verdict=llm_verdict(txt))
            rec['agrees'] = None if rec['llm_verdict'] is None else rec['llm_verdict'] == 'fake'
            if rec['agrees'] is False:
                disagreements.append(f"{rec['face']}, {fmt_range(s['start'], s['end'])}: the timeline flags this segment but the image explainer "
                                     f"judges its key frame ({r['t']:.1f} s) real")
    clean_rec = None
    if clean is not None and clean['frame'] in frames:
        clean_rec = dict(key_frame=dict(frame=clean['frame'], t=clean['t'], p_fake=clean['p_fake']), frame_explanation=None, llm_verdict=None, agrees=None)
        if ask is not None:
            txt = ask(crop_of(frames[clean['frame']], clean), 'Video evidence: no segment of this video is flagged.')
            clean_rec.update(frame_explanation=txt, llm_verdict=llm_verdict(txt))
            clean_rec['agrees'] = None if clean_rec['llm_verdict'] is None else clean_rec['llm_verdict'] == 'real'
            if clean_rec['agrees'] is False and tl['verdict']['label'] == 'real':
                disagreements.append(f"no segment is flagged but the image explainer judges the cleanest frame ({clean['t']:.1f} s) fake")

    notes = []
    sp = tl.get('isolated_spikes') or []
    if sp:
        notes.append(f"{len(sp)} isolated high-score sample{'s' if len(sp) > 1 else ''} ("
                     + ', '.join(f"{x['t']:.1f} s, p {x['p_fake']:.2f}" for x in sp[:5]) + ') did not form a segment and were not counted; worth a look.')
    parts = [summary_text(tl)]
    for rec in out_segments:
        if rec['frame_explanation']:
            parts.append(f"Key frame of {rec['face']}, {fmt_range(rec['start'], rec['end'])} (image explainer): {rec['frame_explanation'].strip()}")
    if clean_rec and clean_rec['frame_explanation']:
        parts.append(f"Cleanest frame ({clean_rec['key_frame']['t']:.1f} s, image explainer): {clean_rec['frame_explanation'].strip()}")
    parts += [f'Note: {n}' for n in notes]
    if disagreements:
        parts.append('Disagreement between the parts of the analysis: ' + '; '.join(disagreements) + '. The timeline verdict stands; treat this video as needing review.')
    return dict(video=header.get('video'), verdict=tl['verdict']['label'], p_video=tl['verdict']['p_video'], summary=summary_text(tl),
                segments=out_segments, clean_frame=clean_rec, notes=notes, disagreements=disagreements, llm_used=ask is not None,
                text='\n\n'.join(parts))


def make_llm_ask(model_path, model_base=None, fp16=True, hint_style='v3', max_new_tokens=200):
    """The image explainer as `ask(image, hint) -> text`, with the same two-step prompt as the image UI: a one-token pass lets the
    heads inside the LLM look at the image, their heatmap hint is built, then the open question is asked with the segment hint in front."""
    from app.engine import M2F2Engine
    from explain.eval_selfexplain import UI_QUESTION, heatmap_hint
    eng = M2F2Engine(model_path, load_4bit=not fp16, model_base=model_base)

    def ask(img, hint):
        eng.ask(img, UI_QUESTION, max_new_tokens=1)
        heat = heatmap_hint(eng.model, img, True, False, hint_style)
        return eng.ask(img, ' '.join(x for x in (hint, heat, UI_QUESTION) if x), max_new_tokens=max_new_tokens)
    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('frames_jsonl')
    ap.add_argument('timeline_json')
    ap.add_argument('--out', required=True, help='JSON output; the text is also written next to it as .txt')
    ap.add_argument('--no-llm', action='store_true', help='rule-based summary only')
    ap.add_argument('--model-path', default=None)
    ap.add_argument('--model-base', default=None)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--hint-style', default='v3', choices=['v2', 'v3'])
    ap.add_argument('--n-key', type=int, default=3)
    a = ap.parse_args()
    ask = None
    if not a.no_llm:
        if not a.model_path:
            ap.error('give --model-path (and --model-base) or use --no-llm')
        ask = make_llm_ask(a.model_path, a.model_base, a.fp16, a.hint_style)
    res = explain_video(a.video, a.frames_jsonl, a.timeline_json, ask, a.n_key)
    json.dump(res, open(a.out, 'w'), indent=1)
    open(os.path.splitext(a.out)[0] + '.txt', 'w').write(res['text'] + '\n')
    print(res['text'])


if __name__ == '__main__':
    main()
