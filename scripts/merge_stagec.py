"""Stage-C training mix: forensic conversations (forensics/texts.py) + DDVQA self-explaining data
(scripts/build_self_explain_data.py), with absolute image paths so --image_folder can be '/'.

python scripts/merge_stagec.py --forensic ../data/stagec/forensic_conversations.json \
    --ddvqa utils/DDVQA_split/c40/train_self_explain_mix.json --ddvqa-images utils/DDVQA_images/c40/train \
    --ddvqa-self 2878 --ddvqa-original 1500 --out ../data/stagec/train_mix.json
"""
import argparse
import json
import os
import random

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--forensic', required=True)
    ap.add_argument('--ddvqa', required=True)
    ap.add_argument('--ddvqa-images', required=True)
    ap.add_argument('--ddvqa-self', type=int, default=2878, help='open-question samples (no label / parts in prompt)')
    ap.add_argument('--ddvqa-original', type=int, default=1500, help='original part-specific conversations')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    forensic = json.load(open(args.forensic))
    dd = json.load(open(args.ddvqa))
    self_exp = list({x['id']: x for x in dd if x['id'].startswith('selfexp_')}.values())    # drop the x2 repeat
    orig = [x for x in dd if not x['id'].startswith('selfexp_')]
    pick = rng.sample(self_exp, min(len(self_exp), args.ddvqa_self)) + rng.sample(orig, min(len(orig), args.ddvqa_original))
    root = os.path.abspath(args.ddvqa_images)
    for x in pick:
        x['image'] = os.path.join(root, x['image'])
    mix = forensic + pick
    rng.shuffle(mix)
    json.dump(mix, open(args.out, 'w'))
    print(f'{len(forensic)} forensic + {len(pick)} DDVQA = {len(mix)} -> {args.out}')
