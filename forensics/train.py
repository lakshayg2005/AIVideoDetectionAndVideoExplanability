"""Stage-A training: M2F2-Det detector + mask decoder + type/filter head (forensics/model.py).

Losses: real/fake CE + type CE + 0.5 filter BCE + forgery mask (BCE + Dice, weighted by mask kind)
+ 0.5 processing mask + 0.5 auxiliary loss on the paper's 24x24 score map.
Sources are sampled evenly. Evaluation per source, on the clean validation set and on a fixed-seed
filtered copy: detection AUC / acc, mask IoU / F1 and pointing game (also for the paper's own map),
false-positive area on real images, type accuracy and filter F1.

python -m forensics.train --index ../data/index/*.jsonl --roots casia=../data/CASIA2 ... \
    --m2f2-shard checkpoints/llava-v1.5-7b-M2F2-Det/model-00003-of-00003.safetensors --out runs/stagea
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from forensics.data import ForgeryDataset, load_index
from forensics.model import M2F2Seg

LOSS_W = {'ce': 1.0, 'type': 1.0, 'filter': 0.5, 'forg': 1.0, 'proc': 0.5, 'aux': 0.5, 'distill': 1.0}


def dice_bce(logits, target, weight):
    """Per-sample BCE + soft Dice; target resized (area) to the logits grid."""
    t = F.interpolate(target, size=logits.shape[-2:], mode='area')
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction='none').flatten(1).mean(1)
    p = torch.sigmoid(logits).flatten(1)
    t = t.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return ((bce + dice) * weight).sum() / weight.sum().clamp(min=1e-6)


def losses(out, b, ablate=(), distill_sources=()):
    w = b['mask_w']
    L = {'ce': F.cross_entropy(out['logits'], b['label']),
         'type': F.cross_entropy(out['type_logits'], b['type']),
         'filter': F.binary_cross_entropy_with_logits(out['filter_logits'], b['filters']),
         'forg': dice_bce(out['mask_logits'][:, :1], b.get('forg_hr', b['forg']), w),
         'proc': dice_bce(out['mask_logits'][:, 1:], b.get('proc_hr', b['proc']), torch.ones_like(w)),
         'aux': dice_bce(out['aux_logits'], b['forg'], w)}
    if 'teacher_logits' in out:    # keep the released detector's decisions on its own domain
        m = torch.tensor([s.startswith(distill_sources) for s in b['source']], device=w.device)
        L['distill'] = (F.kl_div(F.log_softmax(out['logits'][m], -1), F.softmax(out['teacher_logits'][m], -1),
                                 reduction='batchmean') if m.any() else out['logits'].sum() * 0)
    for k in ablate:
        L[k] = L[k] * 0
    return sum(LOSS_W[k] * v for k, v in L.items()), {k: v.item() for k, v in L.items()}


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def collate(items):
    out = {}
    for k in items[0]:
        v = [it[k] for it in items]
        out[k] = torch.stack(v) if torch.is_tensor(v[0]) else (torch.tensor(v) if not isinstance(v[0], str) else v)
    out['mask_w'] = out['mask_w'].float()
    return out


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


@torch.no_grad()
def evaluate(model, loader, dev, size=336):
    model.eval()
    acc = defaultdict(lambda: defaultdict(list))
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
            out = model(b['pixel'], b.get('pixel_hr'))
        p_fake = torch.softmax(out['logits'].float(), -1)[:, 1]
        prob = torch.sigmoid(F.interpolate(out['mask_logits'][:, :1].float(), size=(size, size), mode='bilinear', align_corners=False))
        paper = F.interpolate(out['paper_map'].float(), size=(size, size), mode='bilinear', align_corners=False)
        proc_prob = torch.sigmoid(F.interpolate(out['mask_logits'][:, 1:].float(), size=(size, size), mode='bilinear', align_corners=False))
        for i in range(len(p_fake)):
            g = acc[b['source'][i] + ('/filtered' if b['filtered'][i] > 0 else '')]
            g['y'].append(int(b['label'][i])); g['p'].append(p_fake[i].item())
            g['type_ok'].append(int(out['type_logits'][i].argmax() == b['type'][i]))
            if b['filtered'][i] > 0:
                pf = (out['filter_logits'][i] > 0).float(); tf = b['filters'][i]
                g['filt_tp'].append((pf * tf).sum().item()); g['filt_p'].append(pf.sum().item()); g['filt_t'].append(tf.sum().item())
                inter = ((proc_prob[i, 0] > 0.5) & (b['proc'][i, 0] > 0.5)).sum().item()
                g['proc_iou'].append(inter / max(1, ((proc_prob[i, 0] > 0.5) | (b['proc'][i, 0] > 0.5)).sum().item()))
            gt = b['forg'][i, 0] > 0.5
            pred = prob[i, 0] > 0.5
            if b['label'][i] == 0:
                g['fp_area'].append(pred.float().mean().item())
            elif b['has_region'][i] > 0 and gt.any():
                inter, union = (pred & gt).sum().item(), (pred | gt).sum().item()
                g['iou'].append(inter / max(1, union))
                g['f1'].append(2 * inter / max(1, pred.sum().item() + gt.sum().item()))
                g['point'].append(float(gt.flatten()[prob[i, 0].flatten().argmax()].item()))
                g['point_paper'].append(float(gt.flatten()[paper[i, 0].flatten().argmax()].item()))
    res = {}
    for src, g in sorted(acc.items()):
        y, p = np.array(g['y']), np.array(g['p'])
        r = {'n': len(g['y']), 'auc': auc(g['y'], g['p']), 'acc': float(np.mean((p > 0.5) == y)),
             'fpr': float(np.mean(p[y == 0] > 0.5)) if (y == 0).any() else float('nan'),
             'tpr': float(np.mean(p[y == 1] > 0.5)) if (y == 1).any() else float('nan'),
             'type_acc': float(np.mean(g['type_ok']))}
        for k in ('iou', 'f1', 'point', 'point_paper', 'fp_area', 'proc_iou'):
            if g[k]:
                r[k] = float(np.mean(g[k]))
        if g['filt_t']:
            tp, pp, tt = sum(g['filt_tp']), sum(g['filt_p']), sum(g['filt_t'])
            r['filter_f1'] = 2 * tp / max(1, pp + tt)
        res[src] = r
    model.train()
    return res


def main(args):
    torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roots = dict(kv.split('=', 1) for kv in args.roots)
    train_recs = load_index(args.index, 'train', args.max_per_source, args.seed)
    val_recs = load_index(args.index, args.val_split, args.max_val_per_source, args.seed)
    print(f'train {len(train_recs)}  val {len(val_recs)}', flush=True)

    # source: every (source, label) equally often; type: real images balanced over sources, fakes over forgery
    # types, so rare types (e.g. diffusion face edits) are not a few percent of the samples
    group = (lambda r: (r['source'], r['label'])) if args.balance == 'source' else \
            (lambda r: ('real', r['source']) if r['label'] == 0 else ('fake', r['type']))
    counts = defaultdict(int)
    for r in train_recs:
        counts[group(r)] += 1
    n_groups = {lab: sum(1 for g in counts if (g[1] if args.balance == 'source' else g[0] == 'fake') == lab) for lab in (0, 1)}
    weights = [1.0 / counts[group(r)] / n_groups[r['label']] for r in train_recs]
    print('sampling groups:', {str(k): v for k, v in sorted(counts.items(), key=str)}, flush=True)
    sampler = WeightedRandomSampler(weights, num_samples=args.steps * args.bs, replacement=True)
    train_dl = DataLoader(ForgeryDataset(train_recs, roots, train=True, filter_p=args.filter_p, hr_size=args.hr_size), batch_size=args.bs,
                          sampler=sampler, num_workers=args.workers, collate_fn=collate, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_ds = torch.utils.data.ConcatDataset([
        ForgeryDataset(val_recs, roots, train=False, filter_p=0.0, seed=1, hr_size=args.hr_size),
        ForgeryDataset([r for r in val_recs if r['face']], roots, train=False, filter_p=1.0, seed=2, hr_size=args.hr_size)])
    val_dl = DataLoader(val_ds, batch_size=args.bs, num_workers=args.workers, collate_fn=collate)

    model = M2F2Seg(hires=args.hr_size > 0, use_dino=not args.no_dino)
    print('loaded from M2F2 checkpoint (vision, detector tensors):', model.load_m2f2(args.m2f2_shard), flush=True)
    if args.distill:
        LOSS_W['distill'] = args.distill
        model.make_teacher()
    model.to(dev)
    if dev.type == 'cuda':
        model.clip.half()
        if model.hr is not None and model.hr.dino is not None:
            model.hr.dino.half()
    new = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(('decoder', 'type_head', 'aux', 'hr.', 'refine'))]
    dn = list(model.det.deepfake_encoder.parameters())
    old = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('det.') and 'det.deepfake_encoder.' not in n]
    opt = torch.optim.AdamW([{'params': new, 'lr': args.lr}, {'params': dn, 'lr': args.lr_backbone},
                             {'params': old, 'lr': args.lr_backbone}], weight_decay=1e-4)
    base_lrs = [g['lr'] for g in opt.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == 'cuda')

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'log.jsonl'), 'a')
    step, ckpt = 0, os.path.join(args.out, 'last.pt')
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(s['model'], strict=False); opt.load_state_dict(s['opt']); scaler.load_state_dict(s['scaler'])
        step = s['step']
        print(f'resumed at step {step}', flush=True)
    if step == 0 and args.eval_at_start:
        res = evaluate(model, val_dl, dev)
        log.write(json.dumps({'step': 0, 'eval': res}) + '\n'); log.flush()
        print(json.dumps({'step': 0, 'eval': res}), flush=True)

    ablate = tuple(a for a in args.ablate.split(',') if a)
    model.train()
    model.clip.eval()
    t0, run = time.time(), defaultdict(float)
    t_start = time.time()
    it = iter(train_dl)
    while step < args.steps:
        b = to_dev(next(it), dev)
        lr_f = min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g, lr in zip(opt.param_groups, base_lrs):
            g['lr'] = lr * lr_f
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
            out = model(b['pixel'], b.get('pixel_hr'))
        loss, parts = losses({k: v.float() for k, v in out.items()}, b, ablate, tuple(args.distill_sources))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        for k, v in parts.items():
            run[k] += v
        if step % args.log_every == 0:
            rec = {'step': step, 'sec_per_step': (time.time() - t0) / args.log_every, **{k: v / args.log_every for k, v in run.items()}}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
            t0, run = time.time(), defaultdict(float)
        out_of_time = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes
        if step % args.eval_every == 0 or step == args.steps or out_of_time:
            res = evaluate(model, val_dl, dev)
            print(json.dumps({'step': step, 'eval': res}), flush=True)
            log.write(json.dumps({'step': step, 'eval': res}) + '\n'); log.flush()
            torch.save({'model': model.trainable_state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': step}, ckpt)
        if out_of_time:
            print(f'time budget reached at step {step}', flush=True)
            break


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', nargs='+', required=True)
    ap.add_argument('--roots', nargs='+', required=True, help='name=path for every index root')
    ap.add_argument('--m2f2-shard', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr-backbone', type=float, default=2e-5)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--filter-p', type=float, default=0.3)
    ap.add_argument('--max-per-source', type=int, default=0, help='per (source, label), 0 = all')
    ap.add_argument('--max-val-per-source', type=int, default=300)
    ap.add_argument('--val-split', nargs='+', default=['val', 'test'])
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--eval-at-start', action='store_true')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ablate', default='', help='comma list of loss terms to switch off, e.g. forg,aux')
    ap.add_argument('--max-minutes', type=float, default=0, help='stop (after eval + save) when exceeded')
    ap.add_argument('--distill', type=float, default=0.0, help='weight of distillation from the released detector')
    ap.add_argument('--distill-sources', nargs='+', default=['ddvqa'], help='source prefixes it applies to')
    ap.add_argument('--hr-size', type=int, default=0, help='high-res heatmap branch input size (v5: 672), 0 = off')
    ap.add_argument('--no-dino', action='store_true', help='high-res branch without DINOv2 (ablation)')
    ap.add_argument('--balance', default='source', choices=['source', 'type'])
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
