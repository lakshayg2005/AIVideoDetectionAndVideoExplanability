"""Build the stage-3 training set for a self-explaining model.

DDVQA questions always name the parts to talk about ("Describe ... based on nose, eyebrows."), and the
evaluation additionally states the label. A model trained only that way needs the answer in the prompt
(see RESULTS.md, section 2). Here every annotator's full explanation of an image is turned into
    Q: <open question, no label, no parts>   A: "This image is <label>. <full explanation>"
and mixed with a fraction of the original part-specific conversations, so the model keeps answering
those (used by explain/grounded_explain.py and the DDVQA protocol).

A "full explanation" is the answer to a "Describe ... based on <parts>" question whose part set is not a
strict subset of another question's part set for the same image, i.e. one annotator's complete text
(the single-part questions are derived from it).

python scripts/build_self_explain_data.py --keep-original 0.33
"""
import argparse
import json
import random
import re
from collections import defaultdict

PREFIX = '<image>\n<deepfake>\n'
PARTS_Q = re.compile(r'based on (.+?)\.')
VERDICT = re.compile(r'\s*Based on the learned representation, this image is (real|fake)\.\s*$')
OPEN_QUESTIONS = [
    'Is this face real or fake? Explain why.',
    'Determine the authenticity of this image and explain which facial features support your decision.',
    'Is this image real or fake? Describe the evidence in the face.',
]


def explanations_by_image(data):
    """{image: (label, [full explanations])}"""
    answers, labels = defaultdict(dict), {}    # image -> {frozenset(parts): answer}
    for x in data:
        c = x['conversations']
        for i in range(0, len(c) - 1, 2):
            q, a = c[i]['value'], c[i + 1]['value']
            m = VERDICT.search(a)
            if m:
                labels[x['image']] = m.group(1)
                a = VERDICT.sub('', a)
            m = PARTS_Q.search(q)
            if m and a.strip():
                parts = frozenset(p.strip() for p in m.group(1).split(','))
                answers[x['image']].setdefault(parts, a.strip())
    out = {}
    for img, ans in answers.items():
        full = {a for ps, a in ans.items() if not any(ps < other for other in ans)}
        out[img] = (labels[img], sorted(full))
    return out


def main(args):
    rng = random.Random(args.seed)
    data = json.load(open(args.src))
    expl = explanations_by_image(data)
    self_explain = []
    for img, (label, full) in sorted(expl.items()):
        for k, text in enumerate(full):
            self_explain.append({
                'id': f'selfexp_{img}_{k}', 'image': img,
                'conversations': [
                    {'from': 'human', 'value': PREFIX + rng.choice(OPEN_QUESTIONS)},
                    {'from': 'gpt', 'value': f'This image is {label}. {text}'},
                ],
            })
    original = rng.sample(data, int(len(data) * args.keep_original))
    mix = self_explain * args.repeat + original
    rng.shuffle(mix)
    with open(args.out, 'w') as f:
        json.dump(mix, f)
    n_lab = defaultdict(int)
    for img, (label, _) in expl.items():
        n_lab[label] += 1
    print(f'images {len(expl)} ({dict(n_lab)}), self-explain samples {len(self_explain)} x{args.repeat}, '
          f'original kept {len(original)}/{len(data)}, total {len(mix)} -> {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='./utils/DDVQA_split/c40/train_DDVQA_format.json')
    ap.add_argument('--out', default='./utils/DDVQA_split/c40/train_self_explain_mix.json')
    ap.add_argument('--keep-original', type=float, default=0.33)
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
