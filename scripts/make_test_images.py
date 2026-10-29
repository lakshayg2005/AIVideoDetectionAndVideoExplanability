"""Pick fresh test images the model was never trained on (held-out splits only) and copy them, with ground truth,
into one folder for trying the UI.

Held out from training: OpenForensics Test-Challenge, DiffusionFace LDM / SD-2.1 (unseen generators), DDVQA test,
FakeClue test (all never used for gradient steps); CASIA / SD_inpaint / DiffusionFace `val` (only used to pick
checkpoints). Images identical to anything in --exclude (the existing demo images) are skipped.

python scripts/make_test_images.py --out ../test_images --per 4 --seed 7
"""
import argparse
import hashlib
import json
import os
import random
import shutil

ROOTS = {'ddvqa': './utils', 'casia': '../data/CASIA2', 'sd_inpaint': '../data/SD_inpaint',
         'diffusionface': '../data/DiffusionFace/x', 'fakeclue': '../data/FakeClue', 'openforensics': '../data/OpenForensics'}
# name, index file, filter(record) -> bool
CATS = [
    ('REAL_FF++', 'ddvqa_c40', lambda r: r['split'] == 'test' and r['label'] == 0),
    ('FAKE_faceswap_FF++', 'ddvqa_c40', lambda r: r['split'] == 'test' and r['label'] == 1),
    ('REAL_face_OpenForensics', 'of_challenge', lambda r: r['label'] == 0),
    ('FAKE_face_GANedit_OpenForensics', 'of_challenge', lambda r: r['label'] == 1),
    ('FAKE_diffusion_face_edit', 'diffusionface', lambda r: r['split'] == 'val' and r['image'].startswith('Inpaint/')),
    ('FAKE_diffusion_faceswap', 'diffusionface', lambda r: r['split'] == 'val' and r['image'].startswith('DiffSwap/')),
    ('FAKE_AIgenerated_LDM_unseen', 'diffusionface', lambda r: r['split'] == 'test' and r['image'].startswith('LDM/')),
    ('FAKE_AIgenerated_SD21_unseen', 'diffusionface', lambda r: r['split'] == 'test' and '_2_1_' in r['image']),
    ('FAKE_AIgenerated_people', 'fakeclue', lambda r: r['split'] == 'test' and r['label'] == 1 and r['type'] == 5),
    ('FAKE_photoshop_splice', 'casia', lambda r: r['split'] == 'val' and r['label'] == 1),
    ('REAL_photo_CASIA', 'casia', lambda r: r['split'] == 'val' and r['label'] == 0),
    ('FAKE_AI_inpainting', 'sd_inpaint', lambda r: r['split'] == 'val' and r['label'] == 1),
    ('REAL_photo_SDinpaint', 'sd_inpaint', lambda r: r['split'] == 'val' and r['label'] == 0),
]


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def main(a):
    rng = random.Random(a.seed)
    seen = {md5(os.path.join(a.exclude, f)) for f in os.listdir(a.exclude) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    os.makedirs(a.out, exist_ok=True)
    manifest = []
    for name, idx, ok in CATS:
        recs = [r for r in map(json.loads, open(f'../data/index/{idx}.jsonl')) if ok(r)]
        rng.shuffle(recs)
        n = 0
        for r in recs:
            src = os.path.join(ROOTS[r['root']], r['image'])
            if not os.path.exists(src) or md5(src) in seen:
                continue
            n += 1
            ext = os.path.splitext(src)[1].lower()
            dst = f'{name}_{n}{ext}'
            shutil.copy(src, os.path.join(a.out, dst))
            gt = None
            if r.get('mask') and r['mask_kind'] == 'pixel':
                gt = f'{name}_{n}_gtmask.png'
                shutil.copy(os.path.join(ROOTS[r['root']], r['mask']), os.path.join(a.out, gt))
            manifest.append({'file': dst, 'truth': 'fake' if r['label'] else 'real', 'category': name, 'gt_mask': gt,
                             'source_split': r['split'], 'source_path': r['image']})
            if n == a.per:
                break
        print(f'{name}: {n}')
    json.dump(manifest, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../test_images')
    ap.add_argument('--exclude', default='../demo_images')
    ap.add_argument('--per', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    main(ap.parse_args())
