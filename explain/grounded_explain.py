"""Self-grounded explanation: the model explains its *own* decision on the parts its detector looked at.

The DDVQA protocol (llava/serve/cli_DDVQA_exp.py) puts the ground-truth label and the
human-annotated parts into the prompt ("This image is fake. Describe ... based on nose, eyebrows.").
Here nothing from the ground truth is used:
    verdict  <- the LLM's own real/fake answer (app/engine.py M2F2Engine.detect)
    parts    <- top-k parts of the detector's fused evidence map (explain/detector_maps.py)
    answers  <- the same question templates, filled with the verdict and those parts
The overall answer is written with the DDVQA key "<stem>_0" so eval/eval_explanation.py can score it
against the human answer, i.e. oracle-conditioned vs self-grounded on the same metric.

Ablation: --label-source {model,gt} and --parts-source {map,gt,random} swap in the oracle inputs
one at a time (gt+gt reproduces the DDVQA prompt exactly, as a sanity check).

Also reports text-map consistency: of the facial parts the overall answer mentions, how many are
among the parts the map selected.

python -m explain.grounded_explain --load-4bit
"""
import argparse
import json
import os
import random
import re

from PIL import Image
from tqdm import tqdm

from app.engine import M2F2Engine, build_questions
from explain.detector_maps import DetectorExplainer
from explain.face_parts import PARTS, overlay, part_masks, part_scores

PART_WORDS = {
    'skin': r'\b(skin|face|cheeks?|forehead|chin)\b',
    'eyes': r'\b(eyes?|iris|irises|pupils?|eyelids?)\b',
    'eyebrows': r'\b(eyebrows?|brows?)\b',
    'nose': r'\b(nose|nostrils?)\b',
    'mouth': r'\b(mouth|lips?|teeth)\b',
}


def mentioned_parts(text):
    t = text.lower()
    return {p for p, rx in PART_WORDS.items() if re.search(rx, t)}


def gt_overall(gt_path):
    """{image_stem: (label, [parts])} from the DDVQA overall question, in its original part order."""
    out = {}
    with open(gt_path) as f:
        for line in f:
            (key, v), = json.loads(line).items()
            m = re.match(r'This image is (real|fake)\. Describe the authenticity of the image based on (.+)\.$', v['question'])
            if m and key.endswith('_0'):
                out[key.rsplit('_', 1)[0]] = (m.group(1), [p.strip() for p in m.group(2).split(',')])
    return out


def select_parts(scores, k, rel=0.6):
    """Top-k parts, keeping only those within `rel` of the best part's heat."""
    rank = sorted(scores, key=scores.get, reverse=True)
    best = scores[rank[0]] or 1e-8
    return [p for p in rank[:k] if scores[p] >= rel * best]


def main(args):
    eng = M2F2Engine(args.model_path, load_4bit=args.load_4bit, load_8bit=args.load_8bit)
    ex = DetectorExplainer(eng.model)
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, 'grounded_overlays')
    os.makedirs(vis_dir, exist_ok=True)
    exp_path = os.path.join(args.out_dir, 'grounded_exp_c40.jsonl')
    full_path = os.path.join(args.out_dir, 'grounded_full_c40.jsonl')

    gts = gt_overall(args.gt_path)
    rng = random.Random(0)
    done = set()
    if os.path.exists(full_path):
        with open(full_path) as f:
            done = {json.loads(l)['image'] for l in f if l.strip()}

    files = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith('.jpg'))
    if args.limit:
        files = files[:args.limit]
    for fn in tqdm(files):
        if fn in done:
            continue
        stem = os.path.splitext(fn)[0]
        if 'gt' in (args.label_source, args.parts_source) and stem not in gts:
            continue    # 7 test images have no overall question
        img = Image.open(os.path.join(args.image_dir, fn)).convert('RGB')
        label, p_llm, det_text = eng.detect(img)
        r = ex.explain(img)
        masks, _ = part_masks(img)
        scores = part_scores(r['fused_map'], masks)
        parts = select_parts(scores, args.top_k)
        if args.label_source == 'gt':
            label = gts[stem][0]
        if args.parts_source == 'gt':
            parts = gts[stem][1]
        elif args.parts_source == 'random':
            parts = rng.sample(PARTS, len(parts))

        # per-part templates exist for the 5 PARTS only (GT parts may say "face"); the overall
        # question keeps the given parts verbatim so gt+gt matches the DDVQA prompt exactly
        qs = build_questions(label, [p for p in parts if p in PARTS])
        overall = ('Overall', f"This image is {label}. Describe the authenticity of the image based on {', '.join(parts)}.")
        qs = [overall] + [q for q in qs if q[0] != 'Overall']
        if args.overall_only:
            qs = qs[:1]
        answers = {title: eng.ask(img, q) for title, q in qs}
        said = mentioned_parts(answers.get('Overall', ''))
        rec = {
            'image': fn, 'label_gt': 0 if fn.startswith('5') else 1,
            'verdict': label, 'p_fake_llm': p_llm, 'p_fake_detector': r['p_fake'],
            'parts': parts, 'part_scores': scores, 'answers': answers,
            'mentioned_parts': sorted(said),
            'consistency': (len(said & set(parts)) / len(said)) if said else None,
        }
        with open(full_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        with open(exp_path, 'a') as f:
            f.write(json.dumps({'key': f'{stem}_0', 'image': fn, 'text': answers.get('Overall', '')}) + '\n')
        overlay(img, r['fused_map']).save(os.path.join(vis_dir, f'{stem}_{label}_{"-".join(parts)}.jpg'))

    with open(full_path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    cons = [r['consistency'] for r in recs if r['consistency'] is not None]
    summary = {
        'n': len(recs),
        'verdict_acc': sum((r['verdict'] == 'fake') == (r['label_gt'] == 1) for r in recs) / max(1, len(recs)),
        'llm_detector_agreement': sum((r['verdict'] == 'fake') == (r['p_fake_detector'] >= 0.5) for r in recs) / max(1, len(recs)),
        'text_map_consistency': sum(cons) / len(cons) if cons else None,
        'n_with_mentions': len(cons),
    }
    with open(os.path.join(args.out_dir, 'grounded_summary_c40.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='./checkpoints/llava-v1.5-7b-M2F2-Det')
    ap.add_argument('--load-4bit', action='store_true')
    ap.add_argument('--load-8bit', action='store_true')
    ap.add_argument('--image-dir', default='./utils/DDVQA_images/c40/test')
    ap.add_argument('--out-dir', default='./outputs/explain')
    ap.add_argument('--top-k', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--gt-path', default='./utils/DDVQA_eval/c40/test.jsonl')
    ap.add_argument('--label-source', default='model', choices=['model', 'gt'])
    ap.add_argument('--parts-source', default='map', choices=['map', 'gt', 'random'])
    ap.add_argument('--overall-only', action='store_true')
    main(ap.parse_args())
