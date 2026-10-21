"""Generate "heavy edit" data with exact labels: photo filters applied to real and fake faces.

Filters are processing, not identity / content manipulation, so the content label (real / fake) is
kept and two things are recorded separately:
    edit_mask  - where the filter changed the image (skin only for the local beauty retouch, else all)
    filters    - [{name, strength}] in the order applied
Each sample gets an explanation that names the filters and whether the face itself is authentic, in
the LLaVA conversation format used by train_deepfake.py, plus a metadata record.

python scripts/make_filtered_data.py --src ./utils/DDVQA_images/c40/train --out ../data/filtered/c40_train
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

from forensics.filters import apply, explanation

QUESTION = '<image>\n<deepfake>\nIs this face real or fake? Explain why, including any editing or filters.'


def main(args):
    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'edit_mask'), exist_ok=True)
    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(('.jpg', '.png')))
    if args.limit:
        files = rng.sample(files, min(args.limit, len(files)))
    conv, meta = [], []
    for fn in files:
        label = args.label or ('real' if fn.startswith('5_') else 'fake')    # DDVQA naming
        img = Image.open(os.path.join(args.src, fn)).convert('RGB')
        out, edit, applied = apply(img, rng)
        if not applied:
            continue
        stem = os.path.splitext(fn)[0] + '_' + '-'.join(f['name'] for f in applied)
        out.save(os.path.join(args.out, 'image', stem + '.png'))
        Image.fromarray(edit.astype(np.uint8) * 255).save(os.path.join(args.out, 'edit_mask', stem + '.png'))
        meta.append({'image': stem + '.png', 'source': fn, 'label': label, 'filters': applied})
        conv.append({'id': f'filt_{stem}', 'image': stem + '.png', 'conversations': [
            {'from': 'human', 'value': QUESTION},
            {'from': 'gpt', 'value': explanation(label, applied)}]})
    json.dump(conv, open(os.path.join(args.out, 'conversations.json'), 'w'), indent=1)
    json.dump(meta, open(os.path.join(args.out, 'meta.json'), 'w'), indent=1)
    print(f'{len(meta)} images -> {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', choices=['real', 'fake'], default=None, help='default: DDVQA file naming')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
