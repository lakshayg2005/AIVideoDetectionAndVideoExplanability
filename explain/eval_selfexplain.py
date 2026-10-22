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

QUESTION = 'Is this face real or fake? Explain why.'
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


def verdict(text):
    t = text.lower()
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
        text = eng.ask(img, QUESTION, max_new_tokens=args.max_new_tokens)
        rec = {'source': r['source'], 'image': r['image'], 'label': r['label'], 'type': r['type'],
               'text': text, 'verdict': verdict(text)}
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


def summarize(out_dir):
    rows = [json.loads(l) for l in open(os.path.join(out_dir, 'samples.jsonl'))]
    for x in rows:
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
    args = ap.parse_args()
    summarize(args.out_dir) if args.summarize_only else main(args)
