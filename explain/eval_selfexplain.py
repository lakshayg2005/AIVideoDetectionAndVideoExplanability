"""End-to-end evaluation of a self-explaining model: one open question, no label or parts in the prompt.

For each test image: the model's verdict and explanation; for face images, the parts the text mentions
(explain/grounded_explain.py vocabulary) compared with
  - the parts covered by the model's own predicted forgery mask (forensic_head models): consistency
  - the parts covered by the ground-truth mask: correctness
DDVQA answers are also written as "<stem>_0" so eval/eval_explanation.py scores them against the human
overall answers.

python -m explain.eval_selfexplain --model-path checkpoints/...-lora --model-base checkpoints/... \
    --index ../data/index/ddvqa_c40.jsonl ... --roots ddvqa=./utils ... --out-dir outputs/eval_selfexp/<name>
"""
import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from app.engine import M2F2Engine
from explain.face_parts import PARTS, part_masks
from forensics.data import ForgeryDataset, load_index
from forensics.index import TYPES
from forensics.texts import LIGHTING, NO_REGION_HINT, _position, make_hint

QUESTION = 'Is this face real or fake? Explain why.'
UI_QUESTION = 'Is this image real or fake? Explain why and point out any manipulated region.'    # app/forensic_webui.py
POSITION_WORDS = {'left': r'\bleft\b', 'right': r'\bright\b', 'upper': r'\b(upper|top)\b', 'lower': r'\b(lower|bottom)\b',
                  'centre': r'\b(centre|center|central|middle)\b'}
# stricter than explain/grounded_explain.py: 'face' alone ("this face is fake") is not a mention of the skin
STRICT_PARTS = {
    'skin': r'\b(skin|cheeks?|forehead|chin|complexion)\b',
    'eyes': r'\b(eyes?|iris|irises|pupils?|eyelids?)\b',
    'eyebrows': r'\b(eyebrows?|brows?)\b',
    'nose': r'\b(nose|nostrils?)\b',
    'mouth': r'\b(mouth|lips?|teeth)\b',
}


def mentioned(text):
    t = text.lower()
    return sorted(p for p, rx in STRICT_PARTS.items() if re.search(rx, t))


def covered_parts(mask, img, thr=0.3):
    parts = part_masks(img)[0]
    return sorted(p for p, m in parts.items() if m.any() and (mask & m).sum() / m.sum() > thr)


def positions(text):
    t = text.lower()
    return {w for w, rx in POSITION_WORDS.items() if re.search(rx, t)}


def position_words(pos):
    """forensics.texts._position output ('lower right', 'left', 'centre', ...) -> {'lower', 'right'} etc."""
    return set(pos.split()) - {'middle'} or {'centre'}


def heatmap_hint(model, img, is_face, no_region_any=False, style='v2'):
    """The UI's prompt hint (app/forensic_webui.py _grounding) from the model's own heads; None if they say real.
    The no-region hint is given for faces the type head calls generated / re-rendered (no_region_any: whenever no
    region is marked, the UI's rule before 2026-10-26)."""
    out = model.last_forensics
    if float(torch.softmax(out['logits'], -1)[0, 1]) < 0.5:
        return None
    prob = torch.sigmoid(F.interpolate(out['mask_logits'][:, :1], size=img.size[::-1], mode='bilinear',
                                       align_corners=False))[0, 0].cpu().numpy()
    m = prob > 0.5
    types = torch.softmax(out['type_logits'][0].float(), -1)
    types[TYPES.index('real')] = -1
    t = TYPES[int(types.argmax())]
    if style == 'v3':
        if t in ('img2img', 'generated') or not m.any():
            return make_hint(t)
        if is_face:
            pm, fallback = part_masks(img)
            hit = {p: float((m & pmk).sum() / pmk.sum()) for p, pmk in pm.items() if pmk.any()}
            parts = sorted((p for p, c in hit.items() if c > 0.3), key=lambda p: -hit[p])
            if parts and not fallback:
                return make_hint(t, parts=parts, pct=100 * m.mean())
        return make_hint(t, position=_position(m), pct=100 * m.mean())
    if (is_face and t in ('img2img', 'generated')) or (no_region_any and not m.any()):
        return NO_REGION_HINT
    if not m.any():
        return None
    if is_face:
        pm, fallback = part_masks(img)
        hit = {p: float((m & pmk).sum() / pmk.sum()) for p, pmk in pm.items() if pmk.any()}
        parts = sorted((p for p, c in hit.items() if c > 0.3), key=lambda p: -hit[p])
        if parts and not fallback:
            return f"The forensic heatmap marks the {', '.join(parts)} ({100 * m.mean():.0f}% of the face)."
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() / m.shape[0], xs.mean() / m.shape[1]
    v = 'upper' if cy < 0.33 else 'lower' if cy > 0.66 else 'middle'
    h = 'left' if cx < 0.33 else 'right' if cx > 0.66 else 'centre'
    return f"The forensic heatmap marks the {f'{v} {h}'.replace('middle centre', 'centre')} of the image ({100 * m.mean():.0f}% of it)."


MMTD_REAL = re.compile(r'(has|have) not been tampered|not (been )?tampered with|appears? not to have been tampered|'
                       r'taken directly by (a|the) camera|no (signs|evidence) of tampering')
MMTD_FAKE = re.compile(r'(has|have) been tampered|was tampered|is tampered')


def verdict(text):
    """First real / fake statement; MMTD-style answers ("The picture has (not) been tampered with ...") never
    use those words, so their phrasing is checked first."""
    t = text.lower()
    head = t[:300]
    if MMTD_REAL.search(head):
        return 'real'
    if MMTD_FAKE.search(head):
        return 'fake'
    i_r, i_f = t.find('real'), t.find('fake')
    if i_f < 0 and i_r < 0:
        return None
    return 'fake' if i_r < 0 or (0 <= i_f < i_r) else 'real'


def main(args):
    rng = random.Random(0)
    roots = dict(kv.split('=', 1) for kv in args.roots)
    recs = load_index(args.index, args.split)
    by = {}
    for r in recs:
        by.setdefault((r['source'], r['label']), []).append(r)
    recs = [r for v in by.values() for r in (rng.sample(v, args.per_source) if len(v) > args.per_source else v)]
    ds = ForgeryDataset(recs, roots, train=False)
    eng = M2F2Engine(args.model_path, load_4bit=True, model_base=args.model_base)
    forensic = getattr(eng.model, 'forensic', None) is not None
    os.makedirs(args.out_dir, exist_ok=True)
    full = open(os.path.join(args.out_dir, 'samples.jsonl'), 'w')
    ddvqa = open(os.path.join(args.out_dir, 'ddvqa_exp_c40.jsonl'), 'w')
    for i, r in enumerate(tqdm(recs)):
        img, gt, _, _ = ds.render(i, rng, filter_p=0.0)
        hint = None
        if args.hint:    # as in the UI: heads first, then the open question with their heatmap hint
            eng.ask(img, UI_QUESTION, max_new_tokens=1)
            hint = heatmap_hint(eng.model, img, r['face'], args.hint_old, args.hint_style)
            text = eng.ask(img, f'{hint} {UI_QUESTION}' if hint else UI_QUESTION, max_new_tokens=args.max_new_tokens)
        else:
            text = eng.ask(img, QUESTION, max_new_tokens=args.max_new_tokens)
        rec = {'source': r['source'], 'image': r['image'], 'label': r['label'], 'type': r['type'],
               'text': text, 'verdict': verdict(text), 'hint': hint}
        if r['label'] == 1 and not r['face'] and r['mask_kind'] in ('pixel', 'poly') and gt.any():
            rec['gt_pos'] = _position(gt)
        if r['face']:
            rec['mentioned'] = mentioned(text)
            if r['label'] == 1 and r['mask_kind'] in ('pixel', 'poly', 'weak'):
                rec['gt_parts'] = covered_parts(gt, img)
            if forensic and eng.model.last_forensics is not None:
                m = torch.sigmoid(F.interpolate(eng.model.last_forensics['mask_logits'][:, :1], size=img.size[::-1],
                                                mode='bilinear', align_corners=False))[0, 0].cpu().numpy() > 0.5
                rec['pred_parts'] = covered_parts(m, img)
        full.write(json.dumps(rec) + '\n'); full.flush()
        if r['source'].startswith('ddvqa'):
            stem = os.path.splitext(os.path.basename(r['image']))[0]
            ddvqa.write(json.dumps({'key': f'{stem}_0', 'image': r['image'], 'text': text}) + '\n')
    full.close(); ddvqa.close()
    summarize(args.out_dir)


def gt_positions(args):
    """Ground-truth position of every scene fake's mask, for runs saved before 'gt_pos' was recorded (scene
    images are rendered without randomness, so the masks match those of the run)."""
    rng = random.Random(0)
    roots = dict(kv.split('=', 1) for kv in args.roots)
    recs = load_index(args.index, args.split)
    by = {}
    for r in recs:
        by.setdefault((r['source'], r['label']), []).append(r)
    recs = [r for v in by.values() for r in (rng.sample(v, args.per_source) if len(v) > args.per_source else v)]
    ds = ForgeryDataset(recs, roots, train=False)
    out = {}
    for i, r in enumerate(recs):
        if r['label'] == 1 and not r['face'] and r['mask_kind'] in ('pixel', 'poly'):
            _, gt, _, _ = ds.render(i, rng, filter_p=0.0)
            if gt.any():
                out[r['image']] = _position(gt)
    return out


def summarize(out_dir, gt_pos=None):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, 'samples.jsonl'))]
    for x in rows:
        if gt_pos and 'gt_pos' not in x and x['image'] in gt_pos:
            x['gt_pos'] = gt_pos[x['image']]
    for x in rows:
        x['verdict'] = verdict(x['text'])
        if 'mentioned' in x:
            x['mentioned'] = mentioned(x['text'])
    summary = {}
    for src in sorted({x['source'] for x in rows}):
        rs = [x for x in rows if x['source'] == src]
        s = {'n': len(rs), 'verdict_acc': float(np.mean([x['verdict'] == ('fake' if x['label'] else 'real') for x in rs])),
             'no_verdict': float(np.mean([x['verdict'] is None for x in rs]))}
        for key, name in (('pred_parts', 'consistency_with_own_mask'), ('gt_parts', 'precision_vs_gt_mask')):
            pairs = [(set(x['mentioned']), set(x[key])) for x in rs if x.get(key) is not None and x.get('mentioned')]
            if pairs:
                s[name] = float(np.mean([len(m & p) / len(m) for m, p in pairs]))
                s[name.replace('consistency', 'recall').replace('precision', 'recall')] = float(
                    np.mean([len(m & p) / len(p) if p else 1.0 for m, p in pairs]))
        s['mentions_lighting'] = float(np.mean([bool(LIGHTING.search(x['text'])) for x in rs]))
        s['words'] = float(np.mean([len(x['text'].split()) for x in rs]))
        loc = [(positions(x['text']), position_words(x['gt_pos'])) for x in rs if x.get('gt_pos')]
        if loc:    # scene fakes: are the position words in the answer those of the true region?
            s['location_mentioned'] = float(np.mean([bool(t) for t, _ in loc]))
            said = [(t, g) for t, g in loc if t]
            if said:
                s['location_precision'] = float(np.mean([len(t & g) / len(t) for t, g in said]))
        summary[src] = s
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--model-base', default=None)
    ap.add_argument('--index', nargs='+', required=True)
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--split', nargs='+', default=['test', 'val'])
    ap.add_argument('--per-source', type=int, default=60, help='per (source, label)')
    ap.add_argument('--max-new-tokens', type=int, default=200)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--summarize-only', action='store_true', help='recompute summary.json from samples.jsonl')
    ap.add_argument('--fill-gt-pos', action='store_true', help='with --summarize-only: recompute scene masks (needs '
                    'the run\'s --index / --roots / --per-source) for location_precision of older runs')
    ap.add_argument('--hint', action='store_true', help="UI setting: the heads' heatmap hint and the UI question")
    ap.add_argument('--hint-style', choices=['v2', 'v3'], default='v2', help='v3: hint names the type head\'s verdict')
    ap.add_argument('--hint-old', action='store_true', help='no-region hint whenever no region is marked (UI before 2026-10-26)')
    args = ap.parse_args()
    if args.summarize_only:
        summarize(args.out_dir, gt_positions(args) if args.fill_gt_pos else None)
    else:
        main(args)
