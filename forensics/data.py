"""Dataset for Stage-A training over forensics/index.py records.

Each sample: CLIP-normalised 336x336 image (plain resize, no centre crop, so masks stay aligned),
forgery mask, processing (filter) mask, real/fake label, type, 6 filter labels and a mask weight.
Face boxes are cropped with a margin; DDVQA "weak" masks are the union of the annotated MediaPipe
parts; filters are applied on the fly with exact labels (forensics/filters.py).
"""
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from forensics.filters import FILTERS, apply

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
MASK_WEIGHT = {'pixel': 1.0, 'poly': 1.0, 'full': 1.0, 'none': 1.0, 'weak': 0.3, 'unknown': 0.0}


def load_index(paths, split=None, max_per_source=0, seed=0):
    split = {split} if isinstance(split, str) else split
    recs = []
    for p in paths:
        with open(p) as f:
            recs += [r for r in map(json.loads, f) if split is None or r['split'] in split]
    if max_per_source:
        rng = random.Random(seed)
        by = {}
        for r in recs:
            by.setdefault((r['source'], r['label']), []).append(r)
        recs = [r for v in by.values() for r in (rng.sample(v, max_per_source) if len(v) > max_per_source else v)]
    return recs


def _crop_box(w, h, box, margin=0.3):
    x, y, bw, bh = box
    s = max(bw, bh) * (1 + 2 * margin)
    cx, cy = x + bw / 2, y + bh / 2
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    return x0, y0, int(min(w, cx + s / 2)), int(min(h, cy + s / 2))


class ForgeryDataset(Dataset):
    def __init__(self, records, roots, size=336, train=True, filter_p=0.3, seed=0):
        self.recs, self.roots, self.size = records, roots, size
        self.train, self.filter_p, self.seed = train, filter_p, seed

    def __len__(self):
        return len(self.recs)

    def _forgery_mask(self, r, w, h):
        kind = r['mask_kind']
        if kind == 'pixel':
            m = np.asarray(Image.open(os.path.join(self.roots[r['root']], r['mask'])).convert('L')) > 127
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            return m
        if kind == 'poly':
            m = np.zeros((h, w), np.uint8)
            for p in r['poly']:
                cv2.fillPoly(m, [np.asarray(p, np.int32).reshape(-1, 2)], 1)
            return m > 0
        return np.full((h, w), kind == 'full')

    def render(self, i, rng, filter_p=None):
        """Record i as a size x size PIL image (face crop, plain resize, optional filters) with its forgery
        mask, processing mask and the applied filters."""
        r = self.recs[i]
        filter_p = self.filter_p if filter_p is None else filter_p
        img = Image.open(os.path.join(self.roots[r['root']], r['image'])).convert('RGB')
        w, h = img.size
        forg = self._forgery_mask(r, w, h) if r['mask_kind'] != 'weak' else None
        if r.get('box'):
            x0, y0, x1, y1 = _crop_box(w, h, r['box'])
            img = img.crop((x0, y0, x1, y1))
            forg = forg[y0:y1, x0:x1] if forg is not None else None
        img = img.resize((self.size, self.size), Image.BICUBIC)
        if forg is None:    # weak: union of the annotated parts on the crop
            from explain.face_parts import part_masks
            parts = part_masks(img)[0]
            forg = np.zeros((self.size, self.size), bool)
            for p in r['weak_parts']:
                forg |= parts[p]
        else:
            forg = cv2.resize(forg.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST) > 0

        proc, applied = np.zeros_like(forg), []
        if rng.random() < filter_p:
            allow = FILTERS if r['face'] else [f for f in FILTERS if f != 'beauty']
            img, proc, applied = apply(img, rng, allow=allow)
        return img, forg, proc, applied

    def __getitem__(self, i):
        r = self.recs[i]
        rng = random.Random() if self.train else random.Random(self.seed * 1_000_003 + i)
        img, forg, proc, applied = self.render(i, rng)
        filt = np.zeros(len(FILTERS), np.float32)
        for f in applied:
            filt[FILTERS.index(f['name'])] = 1
        a = np.asarray(img, np.float32) / 255
        if self.train and rng.random() < 0.5:
            a, forg, proc = a[:, ::-1], forg[:, ::-1], proc[:, ::-1]
        a = (a - CLIP_MEAN) / CLIP_STD
        return {
            'pixel': torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))),
            'forg': torch.from_numpy(np.ascontiguousarray(forg[None]).astype(np.float32)),
            'proc': torch.from_numpy(np.ascontiguousarray(proc[None]).astype(np.float32)),
            'label': r['label'], 'type': r['type'], 'filters': torch.from_numpy(filt),
            'mask_w': MASK_WEIGHT[r['mask_kind']], 'has_region': float(r['mask_kind'] in ('pixel', 'poly')),
            'source': r['source'], 'filtered': float(filt.any()),
        }
