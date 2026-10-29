"""Run image folders through the running UI (its /analyze_image API) and report metrics per category.
Truth and category come from the file name (REAL_/FAKE_<category>_<n>.ext).

source ~/miniconda3/bin/activate M2F2_det
python scripts/eval_ui_images.py --dirs ../demo_images ../test_images --out outputs/ui_eval
"""
import argparse
import collections
import json
import os
import re

from gradio_client import Client

EXPECT = [('GANedit', 'face swap'), ('faceswap', 'face swap'), ('face_edit', 'inpainting'), ('inpainting', 'inpainting'),
          ('splice', 'splicing'), ('AIgenerated', 'fully AI-generated')]
NAMED = {'face swap': r'swap|reenact', 'inpainting': r'inpaint|regenerat', 'splicing': r'splic|pasted|copy',
         'fully AI-generated': r'generat|synthetic|rendered'}


def parse_verdict(md):
    m = re.search(r'(LIKELY FAKE|LIKELY REAL|UNCERTAIN — leaning (fake|real))', md)
    if not m:
        return None, False
    return ('fake' if 'FAKE' in m.group(1) or m.group(2) == 'fake' else 'real'), m.group(1).startswith('UNCERTAIN')


def main(a):
    client = Client(a.url, verbose=False)
    os.makedirs(a.out, exist_ok=True)
    for d in a.dirs:
        rows = []
        for f in sorted(os.listdir(d)):
            m = re.match(r'(REAL|FAKE)_(.+?)_\d+\.(png|jpe?g)$', f, re.I)
            if not m or f.endswith('_gtmask.png'):
                continue
            truth, cat = m.group(1).lower(), f'{m.group(1)}_{m.group(2)}'
            md, types, _, _, _, ev, expl, secs = client.predict(os.path.abspath(os.path.join(d, f)), True, True, api_name='/analyze_image')
            verdict, unsure = parse_verdict(md)
            top = types['label'] if types else None
            want = next((w for k, w in EXPECT if k in cat), None) if truth == 'fake' else 'no manipulation'
            rows.append({'file': f, 'category': cat, 'truth': truth, 'verdict': verdict, 'uncertain': unsure, 'type': top,
                         'type_expected': want, 'explanation': expl, 'evidence': ev, 'secs': secs})
            print(f'{f}: {verdict}{"?" if unsure else ""} / {top}', flush=True)
        name = os.path.basename(os.path.normpath(d))
        json.dump(rows, open(os.path.join(a.out, f'{name}.json'), 'w'), indent=1)
        by = collections.defaultdict(list)
        for r in rows:
            by[r['category']].append(r)
        lines = [f'### {name} ({len(rows)} images)', '',
                 '| Category | n | Verdict correct | Uncertain flagged | Type correct | Explanation names type | Mentions lighting |',
                 '|---|---|---|---|---|---|---|']
        pct = lambda v: f'{100 * sum(v) / len(v):.0f}%'
        for cat, rs in by.items():
            ok = [r['verdict'] == r['truth'] for r in rs]
            ty = [r['type_expected'] in (r['type'] or '') for r in rs]
            nm = [bool(re.search(NAMED[r['type_expected']], r['explanation'], re.I)) for r in rs if r['truth'] == 'fake' and r['type_expected'] in NAMED]
            lt = [bool(re.search(r'light|shadow', r['explanation'], re.I)) for r in rs]
            lines.append(f'| {cat} | {len(rs)} | {pct(ok)} | {pct([r["uncertain"] for r in rs])} | {pct(ty)} | '
                         f'{pct(nm) if nm else "-"} | {pct(lt)} |')
        for t in ('fake', 'real'):
            rs = [r for r in rows if r['truth'] == t]
            if rs:
                lines.append(f'| **all {t}** | {len(rs)} | {pct([r["verdict"] == t for r in rs])} | | | | |')
        lines.append(f'| **overall** | {len(rows)} | {pct([r["verdict"] == r["truth"] for r in rows])} | | | | |')
        open(os.path.join(a.out, f'{name}.md'), 'w').write('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:7861')
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out', default='outputs/ui_eval')
    main(ap.parse_args())
