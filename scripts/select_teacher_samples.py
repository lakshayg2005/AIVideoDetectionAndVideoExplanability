"""Pick training images for teacher descriptions (explain/teacher_annotate.py) and record their ground-truth region.

Tasks:
  ai_face       fully generated / re-rendered faces (DiffusionFace) and AI-generated people (FakeClue Chameleon):
                image-specific artifacts instead of a template sentence
  face_edit     DiffusionFace inpainting edits: what the edited region shows (edit prompt known)
  splice        CASIA v2: what exactly was pasted / copied (e.g. "the cat's face", not "the cat")
  scene_inpaint SD_inpaint: what the regenerated region shows
Only training splits are used. Output: JSON list with absolute image path, task, box (x0, y0, x1, y1 in image
pixels, from the ground-truth mask) and edit prompt.

python scripts/select_teacher_samples.py --out ../data/teacher/samples.json
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

ROOTS = {'diffusionface': '../data/DiffusionFace/x', 'casia': '../data/CASIA2', 'sd_inpaint': '../data/SD_inpaint',
         'fakeclue': '../data/FakeClue'}
INDEX = '../data/index'


def box_of(mask_path, size):
    m = Image.open(mask_path).convert('L')
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    ys, xs = np.nonzero(np.asarray(m) > 127)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def main(args):
    rng = random.Random(args.seed)
    load = lambda f: [json.loads(l) for l in open(os.path.join(INDEX, f)) if '"split": "train"' in l]
    df, casia, sd, fc = load('diffusionface.jsonl'), load('casia.jsonl'), load('sd_inpaint.jsonl'), load('fakeclue.jsonl')
    pick = lambda rs, n: rng.sample(rs, min(n, len(rs)))
    plan = [
        ('ai_face', pick([r for r in df if r['label'] and r['mask_kind'] == 'full'], args.n_ai_face)),
        ('ai_face', pick([r for r in fc if r['label'] and r['image'].startswith('train/chameleon')], args.n_chameleon)),
        ('face_edit', pick([r for r in df if r['label'] and r['mask_kind'] == 'pixel'], args.n_face_edit)),
        ('splice', pick([r for r in casia if r['label']], args.n_splice)),
        ('scene_inpaint', pick([r for r in sd if r['label']], args.n_scene_inpaint)),
    ]
    out = []
    for task, rs in plan:
        for r in rs:
            path = os.path.abspath(os.path.join(ROOTS[r['root']], r['image']))
            box = None
            if r.get('mask'):
                with Image.open(path) as im:
                    box = box_of(os.path.join(ROOTS[r['root']], r['mask']), im.size)
                if box is None:
                    continue
            out.append({'key': f"{r['root']}/{r['image']}", 'image': path, 'task': task, 'box': box,
                        'edit': r.get('edit') or None})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print(len(out), 'samples', {t: sum(o['task'] == t for o in out) for t in ('ai_face', 'face_edit', 'splice', 'scene_inpaint')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../data/teacher/samples.json')
    # ai_face is off by default: on 16 test images the 3B teacher gave every AI face the same "unnaturally smooth
    # hair" sentence, a template like MMTD's lighting one. FakeClue's descriptions cover AI-generated images instead.
    ap.add_argument('--n-ai-face', type=int, default=0)
    ap.add_argument('--n-chameleon', type=int, default=0)
    ap.add_argument('--n-face-edit', type=int, default=800)
    ap.add_argument('--n-splice', type=int, default=1000)
    ap.add_argument('--n-scene-inpaint', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
