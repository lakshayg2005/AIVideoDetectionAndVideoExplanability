"""Unified index of forgery data for Stage-A training (detector + mask head + type head).

One JSON record per sample, paths relative to a named root so the same index works locally and on
Kaggle (roots are passed to the dataset at load time):
    root, image, label (0 real / 1 fake), type (TYPES index), face (bool), source, split,
    mask_kind: pixel | poly | full | none | weak
    mask (pixel), poly (COCO polygons, image coordinates), box ([x, y, w, h] face box), weak_parts

python -m forensics.index casia --root ../data/CASIA2 --out ../data/index/casia.jsonl
"""
import argparse
import json
import os
import random
import re

TYPES = ['real', 'face_swap', 'splice', 'diffusion_local', 'img2img', 'generated']
PARTS = ['skin', 'eyes', 'eyebrows', 'nose', 'mouth']


def _split(key, val_frac, seed=0):
    return 'val' if random.Random(f'{seed}-{key}').random() < val_frac else 'train'


def casia(root, val_frac=0.1):
    """CASIA v2 (Kaggle copy) + revised ground truth (github.com/namtpham/casia2groundtruth)."""
    gt_dir = 'gt/CASIA2.0_Groundtruth'
    gts = {f[:-len('_gt.png')] for f in os.listdir(os.path.join(root, gt_dir)) if f.endswith('_gt.png')}
    for sub, label in (('Au', 0), ('Tp', 1)):
        for f in sorted(os.listdir(os.path.join(root, 'CASIA2', sub))):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in ('.jpg', '.tif', '.bmp', '.png') or (label and stem not in gts):
                continue
            yield {'root': 'casia', 'image': f'CASIA2/{sub}/{f}', 'label': label, 'type': TYPES.index('splice') if label else 0,
                   'face': False, 'source': 'casia', 'split': _split(stem, val_frac),
                   'mask_kind': 'pixel' if label else 'none', 'mask': f'{gt_dir}/{stem}_gt.png' if label else None}


def sd_inpaint(root):
    """FakeShield SD_inpaint_dataset: {train,eval}/Au/image, {train,eval}/Tp/{image,mask}."""
    for part, split in (('train', 'train'), ('eval', 'val')):
        for sub, label in (('Au', 0), ('Tp', 1)):
            d = os.path.join(root, part, sub, 'image')
            masks = set(os.listdir(os.path.join(root, part, sub, 'mask'))) if label else set()
            for f in sorted(os.listdir(d)):
                m = os.path.splitext(f)[0] + '.png'
                if label and m not in masks:
                    continue
                yield {'root': 'sd_inpaint', 'image': f'{part}/{sub}/image/{f}', 'label': label,
                       'type': TYPES.index('diffusion_local') if label else 0, 'face': False, 'source': 'sd_inpaint',
                       'split': split, 'mask_kind': 'pixel' if label else 'none',
                       'mask': f'{part}/{sub}/mask/{m}' if label else None}


def _parts(text):
    return sorted({'skin' if p.strip() == 'face' else p.strip() for p in text.split(',')} & set(PARTS))


def ddvqa(root, comp='c40'):
    """DDVQA face crops (repo utils/). Fakes get weak masks from the annotated parts."""
    weak = {}
    for x in json.load(open(os.path.join(root, f'DDVQA_split/{comp}/train_DDVQA_format.json'))):
        for c in x['conversations']:
            m = re.search(r'based on (.+?)\.', c['value'])
            if m:
                weak.setdefault(x['image'], set()).update(_parts(m.group(1)))
    with open(os.path.join(root, f'DDVQA_eval/{comp}/test.jsonl')) as f:
        for line in f:
            (key, v), = json.loads(line).items()
            m = re.search(r'This image is fake\. Describe the authenticity of the image based on (.+)\.', v['question'])
            if m:
                weak.setdefault(key.rsplit('_', 1)[0] + '.jpg', set()).update(_parts(m.group(1)))
    for part, split in (('train', 'train'), ('test', 'test')):
        for f in sorted(os.listdir(os.path.join(root, f'DDVQA_images/{comp}/{part}'))):
            label = 0 if f.startswith('5_') else 1
            parts = sorted(weak.get(f, ())) if label else []
            yield {'root': 'ddvqa', 'image': f'DDVQA_images/{comp}/{part}/{f}', 'label': label,
                   'type': TYPES.index('face_swap') if label else 0, 'face': True, 'source': f'ddvqa_{comp}',
                   'split': split, 'mask_kind': 'weak' if parts else 'none', 'weak_parts': parts}


def openforensics(root, poly_json, split, min_side=48, val_frac=0.0):
    """OpenForensics (COCO polygons): one record per face, cropped by its box at load time."""
    d = json.load(open(os.path.join(root, poly_json)))
    images = {im['id']: im for im in d['images']}
    for a in d['annotations']:
        x, y, w, h = a['bbox']
        if min(w, h) < min_side:
            continue
        im = images[a['image_id']]
        seg = a['segmentation']
        seg = json.loads(seg) if isinstance(seg, str) else seg
        label = int(a['category_id'])
        s = _split(im['id'], val_frac) if val_frac else split
        yield {'root': 'openforensics', 'image': im['file_name'], 'label': label,
               'type': TYPES.index('face_swap') if label else 0, 'face': True, 'source': 'openforensics',
               'split': s, 'box': [x, y, w, h], 'mask_kind': 'poly' if label else 'none', 'poly': seg if label else None}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('source', choices=['casia', 'sd_inpaint', 'ddvqa', 'openforensics'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--poly-json', default='Val_poly.json')
    ap.add_argument('--split', default='train')
    ap.add_argument('--val-frac', type=float, default=0.0, help='openforensics: split one file by image')
    ap.add_argument('--comp', default='c40')
    args = ap.parse_args()
    gen = {'casia': lambda: casia(args.root), 'sd_inpaint': lambda: sd_inpaint(args.root),
           'ddvqa': lambda: ddvqa(args.root, args.comp),
           'openforensics': lambda: openforensics(args.root, args.poly_json, args.split, val_frac=args.val_frac)}[args.source]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    n, stats = 0, {}
    with open(args.out, 'w') as f:
        for r in gen():
            f.write(json.dumps(r) + '\n')
            n += 1
            k = (r['split'], r['label'])
            stats[k] = stats.get(k, 0) + 1
    print(f'{args.source}: {n} records -> {args.out}', {f'{s}/{"fake" if l else "real"}': c for (s, l), c in sorted(stats.items())})
