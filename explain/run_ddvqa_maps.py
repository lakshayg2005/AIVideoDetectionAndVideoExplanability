"""Grounding + faithfulness evaluation of M2F2-Det's detector maps on DDVQA test.

For every test image:
  1. detector p_fake and the patch / Grad-CAM / fused maps (explain/detector_maps.py)
  2. per-part heat (skin, eyes, eyebrows, nose, mouth) and the part ranking
  3. faithfulness: blur the top-ranked part vs. a random part vs. the lowest part, record drop in p_fake

Part localization is scored on fake images against the human-annotated parts in
utils/DDVQA_eval/c40/test.jsonl ("... based on nose, eyebrows."), with a random-ranking baseline.

python -m explain.run_ddvqa_maps --model-path ./checkpoints/llava-v1.5-7b-M2F2-Det --load-4bit
"""
import argparse
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from explain.detector_maps import DetectorExplainer
from explain.face_parts import PARTS, occlude, overlay, part_masks, part_scores
from llava.model.builder import load_deepfake_model
from llava.mm_utils import get_model_name_from_path

MAPS = ['patch_map', 'gradcam_map', 'fused_map']


def gt_parts(gt_path):
    """{image_stem: set(parts)} from the overall 'based on ...' question (fake images only)."""
    out = {}
    with open(gt_path) as f:
        for line in f:
            (key, v), = json.loads(line).items()
            m = re.search(r'This image is fake\. Describe the authenticity of the image based on (.+)\.', v['question'])
            if m:
                parts = {p.strip().replace('face', 'skin') for p in m.group(1).split(',')}
                out[key.rsplit('_', 1)[0]] = parts & set(PARTS)
    return out


def ranking_hits(rank, gt, k):
    return float(any(p in gt for p in rank[:k]))


def main(args):
    random.seed(0)
    np.random.seed(0)
    name = get_model_name_from_path(args.model_path)
    _, model, _, _ = load_deepfake_model(args.model_path, args.model_base, name, args.load_8bit, args.load_4bit, device='cuda')
    model.eval()
    if not (args.load_4bit or args.load_8bit):
        model = model.to('cuda')
    model.load_deepfake_encoder(model.config.deepfake_model_path, verbose=True)
    model.deepfake_encoder.to('cuda', dtype=torch.float16)
    ex = DetectorExplainer(model)

    gts = gt_parts(args.gt_path)
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, 'overlays')
    os.makedirs(vis_dir, exist_ok=True)
    rec_path = os.path.join(args.out_dir, 'maps_c40.jsonl')

    files = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith('.jpg'))
    if args.limit:
        files = files[:args.limit]
    records = []
    for fn in tqdm(files):
        img = Image.open(os.path.join(args.image_dir, fn)).convert('RGB')
        stem = os.path.splitext(fn)[0]
        label = 0 if fn.startswith('5') else 1
        r = ex.explain(img)
        masks, fallback = part_masks(img)
        rec = {'image': fn, 'label': label, 'p_fake': r['p_fake'], 'branch': r['branch'],
               'mask_fallback': fallback, 'gt_parts': sorted(gts.get(stem, []))}
        for mname in MAPS:
            s = part_scores(r[mname], masks)
            rec[mname] = {'part_scores': s, 'rank': sorted(s, key=s.get, reverse=True)}

        rank = rec['fused_map']['rank']
        p0, m0 = ex.score(img)
        valid = [p for p in rank if masks[p].any()]
        rnd = random.choice(valid[1:]) if len(valid) > 1 else valid[0]
        rec['faith'] = {'p_orig': p0, 'margin_orig': m0}
        for name, part in (('top', valid[0]), ('random', rnd), ('bottom', valid[-1])):
            p, m = ex.score(occlude(img, masks[part], mode=args.occlude_mode))
            rec['faith'][f'drop_{name}'] = m0 - m    # drop in fake-real logit margin
            rec['faith'][f'pdrop_{name}'] = p0 - p
        records.append(rec)
        with open(rec_path, 'a' if len(records) > 1 else 'w') as f:
            f.write(json.dumps(rec) + '\n')

        if args.save_overlays:
            panel = Image.new('RGB', (img.width * 3, img.height))
            panel.paste(img, (0, 0))
            panel.paste(overlay(img, r['patch_map']), (img.width, 0))
            panel.paste(overlay(img, r['gradcam_map']), (img.width * 2, 0))
            panel.save(os.path.join(vis_dir, f'{stem}_p{r["p_fake"]:.2f}_{rank[0]}.jpg'))

    summary = summarize(records)
    with open(os.path.join(args.out_dir, 'summary_c40.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def summarize(records):
    y = np.array([r['label'] for r in records])
    p = np.array([r['p_fake'] for r in records])
    s = {'n': len(records), 'detector_acc@0.5': float(((p > 0.5) == y).mean()),
         'mask_fallback_rate': float(np.mean([r['mask_fallback'] for r in records]))}
    try:
        from sklearn.metrics import roc_auc_score
        s['detector_auc'] = float(roc_auc_score(y, p))
    except Exception:
        pass

    fakes = [r for r in records if r['label'] == 1 and r['gt_parts']]
    s['n_fake_with_gt_parts'] = len(fakes)
    if fakes:
        rng = np.random.default_rng(0)
        for k in (1, 2):
            for mname in MAPS:
                s[f'{mname}_hit@{k}'] = float(np.mean([ranking_hits(r[mname]['rank'], set(r['gt_parts']), k) for r in fakes]))
            s[f'random_hit@{k}'] = float(np.mean([
                np.mean([ranking_hits(list(rng.permutation(PARTS)), set(r['gt_parts']), k) for _ in range(200)])
                for r in fakes]))

    for group, sel in (('fake', y == 1), ('all', np.ones_like(y, bool))):
        rs = [r for r, keep in zip(records, sel) if keep]
        for key in ('drop_top', 'drop_random', 'drop_bottom', 'pdrop_top', 'pdrop_random', 'pdrop_bottom'):
            s[f'faith_{group}_{key}'] = float(np.mean([r['faith'][key] for r in rs]))
        # how often blurring the top-ranked part hurts the decision more than blurring a random part
        s[f'faith_{group}_top_gt_random'] = float(np.mean([r['faith']['drop_top'] > r['faith']['drop_random'] for r in rs]))
    return s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='./checkpoints/llava-v1.5-7b-M2F2-Det')
    ap.add_argument('--model-base', default=None)
    ap.add_argument('--load-4bit', action='store_true')
    ap.add_argument('--load-8bit', action='store_true')
    ap.add_argument('--image-dir', default='./utils/DDVQA_images/c40/test')
    ap.add_argument('--gt-path', default='./utils/DDVQA_eval/c40/test.jsonl')
    ap.add_argument('--out-dir', default='./outputs/explain')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-overlays', action='store_true')
    ap.add_argument('--occlude-mode', default='blur', choices=['blur', 'mean'])
    main(ap.parse_args())
