"""Starting checkpoint for stages B / C with the Stage-A forensic heads (PLAN.md).

Released LLaVA-M2F2 checkpoint + Stage-A detector and heads (forensics/train.py output):
  - shards without deepfake_* tensors are symlinked; the shard holding them is rewritten without them
  - extra shard: deepfake_encoder.* = Stage-A detector (text-encoder weights from the release, they are
    frozen in Stage A), forensic.* = mask decoder / aux / type head, deepfake_projector.* = projector for the
    EVIDENCE_DIM evidence vector, initialised from the released 2-dim projector on the real/fake inputs and
    zero on the new ones (--random-projector: std 0.1 as merge_lora_weights_deepfake_random.py)
  - config: forensic_head = true, deepfake_model_path cleared so no .pth overrides the Stage-A detector

python scripts/prepare_forensic_base.py --m2f2-path checkpoints/llava-v1.5-7b-M2F2-Det \
    --stagea runs/stagea_v2/last.pt --out checkpoints/llava-v1.5-7b-M2F2-forensic-base
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from forensics.model import EVIDENCE_DIM


def main(args):
    os.makedirs(args.out, exist_ok=True)
    idx = json.load(open(os.path.join(args.m2f2_path, 'model.safetensors.index.json')))
    wm = idx['weight_map']
    drop = lambda k: k.startswith(('deepfake_encoder.', 'deepfake_projector.'))
    dirty = {f for k, f in wm.items() if drop(k)}
    for src in glob.glob(os.path.join(args.m2f2_path, '*')):
        name = os.path.basename(src)
        if name in dirty or name in ('config.json', 'model.safetensors.index.json'):
            continue
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    text_enc, old_proj = {}, {}
    for f in sorted(dirty):
        keep = {}
        with safe_open(os.path.join(args.m2f2_path, f), framework='pt') as sf:
            for k in sf.keys():
                if not drop(k):
                    keep[k] = sf.get_tensor(k)
                elif k.startswith('deepfake_encoder.clip_text_encoder.model.'):
                    text_enc[k] = sf.get_tensor(k)
                elif k.startswith('deepfake_projector.'):
                    old_proj[k[len('deepfake_projector.'):]] = sf.get_tensor(k)
        save_file(keep, os.path.join(args.out, f), metadata={'format': 'pt'})
        print(f'rewrote {f}: {len(keep)} tensors')

    sa = torch.load(args.stagea, map_location='cpu')['model']
    new = dict(text_enc)
    for k, v in sa.items():
        if k.startswith('det.'):
            new['deepfake_encoder.' + k[4:]] = v
        elif k.startswith(('decoder.', 'type_head.')) or k == 'aux':
            new['forensic.' + k] = v
    torch.manual_seed(0)
    hidden = json.load(open(os.path.join(args.m2f2_path, 'config.json')))['hidden_size']
    proj = torch.nn.Sequential(torch.nn.Linear(EVIDENCE_DIM, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden))
    if args.random_projector:
        for m in proj.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.1)
    else:
        # function-preserving: the real/fake inputs (evidence[:2]) reuse the released 2-dim projector and the new
        # inputs start at zero, so at step 0 the evidence token equals the released model's <deepfake> token
        with torch.no_grad():
            proj[0].weight.zero_()
            proj[0].weight[:, :2] = old_proj['0.weight'].float()
            proj[0].bias.copy_(old_proj['0.bias'].float())
            proj[2].weight.copy_(old_proj['2.weight'].float())
            proj[2].bias.copy_(old_proj['2.bias'].float())
    new.update({f'deepfake_projector.{k}': v for k, v in proj.state_dict().items()})
    new = {k: v.half().contiguous() for k, v in new.items()}
    save_file(new, os.path.join(args.out, 'model-forensic.safetensors'), metadata={'format': 'pt'})
    print(f'wrote model-forensic.safetensors: {len(new)} tensors '
          f'({sum(k.startswith("forensic.") for k in new)} forensic, {sum(k.startswith("deepfake_encoder.") for k in new)} detector)')

    wm = {k: f for k, f in wm.items() if not drop(k)}
    wm.update({k: 'model-forensic.safetensors' for k in new})
    json.dump({'metadata': {}, 'weight_map': wm}, open(os.path.join(args.out, 'model.safetensors.index.json'), 'w'), indent=2)
    cfg = json.load(open(os.path.join(args.m2f2_path, 'config.json')))
    cfg.update({'forensic_head': True, 'forensic_evidence_dim': EVIDENCE_DIM, 'deepfake_model_path': None})
    json.dump(cfg, open(os.path.join(args.out, 'config.json'), 'w'), indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--m2f2-path', default='./checkpoints/llava-v1.5-7b-M2F2-Det')
    ap.add_argument('--stagea', default='./runs/stagea_v2/last.pt')
    ap.add_argument('--out', default='./checkpoints/llava-v1.5-7b-M2F2-forensic-base')
    ap.add_argument('--random-projector', action='store_true', help='random init instead of function-preserving')
    main(ap.parse_args())
