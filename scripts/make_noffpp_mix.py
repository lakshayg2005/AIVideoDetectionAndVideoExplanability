"""Stage-C text mix without any FaceForensics++-derived sample (FF++ is kept for testing, with ExDDV).

Drops DDVQA conversations (ids '<num>' / 'selfexp_*', FF++ frames) and 'fakeclue_ffpp_*'; keeps the forensic
conversations of CASIA, SD_inpaint, OpenForensics, DiffusionFace and FakeClue (AI-generated people). Image paths of the
old machine are rewritten to the new data root.
python scripts/make_noffpp_mix.py --src data/stagec_v3/train_mix.json --out data/stagec_v7/train_mix.json
"""
import argparse, json, os

KEEP = ('diffusionface_', 'sd_inpaint_', 'casia_', 'openforensics_', 'fakeclue_human_')
OLD = '/home/cse/Desktop/Lakshay/BTP-1/data/'

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--data-root', default='/data3/Aditya/bp_data/data/')
a = ap.parse_args()
d = json.load(open(a.src))
keep, drop = [], 0
for x in d:
    if not x['id'].startswith(KEEP) or 'ff++' in x['image'].lower() or 'ddvqa' in x['image'].lower():
        drop += 1
        continue
    x['image'] = x['image'].replace(OLD, a.data_root)
    keep.append(x)
missing = [x['image'] for x in keep if not os.path.exists(x['image'])]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(keep, open(a.out, 'w'))
print(f'kept {len(keep)}, dropped {drop} (DDVQA / FF++), missing image files: {len(missing)}', missing[:3])
