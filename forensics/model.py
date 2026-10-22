"""M2F2-Det detector extended with a forgery/processing mask decoder and an edit-type + filter head.

Kept from the paper (weights from the released checkpoint): the LLaVA CLIP ViT-L/14-336 vision tower
(frozen, fine-tuned by M2F2), the forgery-prompt text branch, DenseNet-121 and the real/fake output.
New:
  MaskDecoder - fuses CLIP patch tokens of three layers, the 24x24 forgery-prompt score map and
                DenseNet features (84/42/21 px grids) -> 2 x 84x84 logits: forgery, processing
  aux map     - affine read-out of the paper's 24x24 score map, supervised with the forgery mask
  TypeHead    - [CLS, DenseNet feature, patch features pooled under the predicted mask] ->
                6 forgery types + 6 filters
  teacher     - optional frozen copy of the released detector; its real/fake logits are distilled on
                the paper's own domain (FF++ / DDVQA) so fine-tuning on other data does not erode it
"""
import copy
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from transformers import CLIPVisionModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llava', 'model', 'deepfake'))
from M2F2Det.model import M2F2Det  # noqa: E402  (imported without the LLaVA package)

CLIP_NAME = 'openai/clip-vit-large-patch14-336'
CLIP_LAYERS = (8, 16, -2)
N_TYPES, N_FILTERS = 6, 6
EVIDENCE_DIM = 2 + N_TYPES + N_FILTERS + 2    # real/fake, types, filters, forged / filtered area


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class MaskDecoder(nn.Module):
    def __init__(self, clip_dim=1024, n_clip=len(CLIP_LAYERS), mid=128, out=2):
        super().__init__()
        self.clip_in = nn.ModuleList([nn.Conv2d(clip_dim, mid, 1) for _ in range(n_clip)])
        self.top = block(mid * n_clip + 1, 256)
        self.f3, self.f2, self.f1 = nn.Conv2d(1024, 128, 1), nn.Conv2d(512, 128, 1), nn.Conv2d(256, 64, 1)
        self.fuse42 = block(256 + 128 + 128, 192)
        self.fuse84 = block(192 + 64, 96)
        self.head = nn.Conv2d(96, out, 1)

    def forward(self, clip_maps, score_map, f1, f2, f3):
        x = self.top(torch.cat([c(m) for c, m in zip(self.clip_in, clip_maps)] + [score_map], 1))    # 24x24
        up = lambda t, s: F.interpolate(t, size=s, mode='bilinear', align_corners=False)
        x = self.fuse42(torch.cat([up(x, f2.shape[-2:]), self.f2(f2), up(self.f3(f3), f2.shape[-2:])], 1))
        x = self.fuse84(torch.cat([up(x, f1.shape[-2:]), self.f1(f1)], 1))
        return self.head(x)    # [B, 2, 84, 84]


def densenet(det, x):
    """DenseNet-121 of the M2F2 detector with the 84 / 42 / 21 px block outputs (336 px input)."""
    feats = {}
    for name, m in det.deepfake_encoder.named_children():
        x = m(x)
        if name in ('denseblock1', 'denseblock2', 'denseblock3'):
            feats[name] = x
    return feats['denseblock1'], feats['denseblock2'], feats['denseblock3'], x


def run_heads(det, decoder, aux, type_head, pixel, hs, text=None):
    """Detector + new heads. hs: CLIP hidden states of CLIP_LAYERS (float), pixel: CLIP-normalised 336 px.
    text: optional cached forgery-prompt features (det.text_proj(det.clip_text_encoder()))."""
    B = pixel.shape[0]
    v = det.vision_proj(hs[-1])
    cls, patches = v[:, 0], v[:, 1:]
    text = det.text_proj(det.clip_text_encoder().float()) if text is None else text
    scores = F.cosine_similarity(patches, text.unsqueeze(1).expand_as(patches), dim=-1)    # [B, 576]
    side = int(scores.shape[1] ** 0.5)
    f1, f2, f3, f4 = densenet(det, pixel)
    dfeat = det.deepfake_proj(f4.mean(dim=(2, 3)))
    logits = det.output(torch.cat([det.clip_text_alpha * scores, det.clip_vision_alpha * cls, dfeat], -1))

    score_map = scores.view(B, 1, side, side)
    clip_maps = [h[:, 1:].transpose(1, 2).reshape(B, -1, side, side) for h in hs]
    mask_logits = decoder(clip_maps, score_map, f1, f2, f3)
    aux_logits = aux[0] * score_map + aux[1]

    wgt = torch.sigmoid(F.interpolate(mask_logits[:, :1], size=(side, side), mode='bilinear', align_corners=False))
    wgt = wgt.flatten(1).unsqueeze(-1) + 1e-4
    pooled = (patches * wgt).sum(1) / wgt.sum(1)
    th = type_head(torch.cat([cls, dfeat, pooled], -1))
    w = det.output.weight[1] - det.output.weight[0]    # the paper's map: exact patch contribution to the margin
    return {'logits': logits, 'mask_logits': mask_logits, 'aux_logits': aux_logits,
            'type_logits': th[:, :N_TYPES], 'filter_logits': th[:, N_TYPES:],
            'paper_map': (w[:scores.shape[1]] * det.clip_text_alpha * scores).view(B, 1, side, side)}


def evidence(out, hs_last):
    """Inputs for the LLM: [B, EVIDENCE_DIM] evidence vector and [B, 3, 1024] CLIP patch features pooled in the
    predicted forged area, outside it, and in the filtered area (hs_last: CLIP layer -2 incl. CLS)."""
    side = out['paper_map'].shape[-1]
    m = torch.sigmoid(F.interpolate(out['mask_logits'].float(), size=(side, side), mode='bilinear', align_corners=False))
    forg, proc = m[:, 0].flatten(1), m[:, 1].flatten(1)
    ev = torch.cat([torch.softmax(out['logits'].float(), -1), torch.softmax(out['type_logits'].float(), -1),
                    torch.sigmoid(out['filter_logits'].float()), forg.mean(1, keepdim=True), proc.mean(1, keepdim=True)], -1)
    patches = hs_last[:, 1:].float()

    def pool(w):
        w = w.unsqueeze(-1) + 1e-4
        return (patches * w).sum(1) / w.sum(1)
    return ev, torch.stack([pool(forg), pool(1 - forg), pool(proc)], 1)


class ForensicHeads(nn.Module):
    """The new Stage-A heads as one module, for use inside the LLaVA model (which owns the vision tower and
    the detector as `deepfake_encoder`). Parameter names match M2F2Seg so its checkpoints map directly."""

    def __init__(self):
        super().__init__()
        self.decoder = MaskDecoder()
        self.aux = nn.Parameter(torch.tensor([10.0, 0.0]))
        self.type_head = nn.Sequential(nn.Linear(3 * 1024, 512), nn.GELU(), nn.Dropout(0.1),
                                       nn.Linear(512, N_TYPES + N_FILTERS))

    def forward(self, det, pixel, hs, text=None):
        return run_heads(det, self.decoder, self.aux, self.type_head, pixel, hs, text)


class M2F2Seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(CLIP_NAME)    # .half() on GPU (train.py)
        self.clip.requires_grad_(False).eval()
        self.det = M2F2Det(clip_text_encoder_name=CLIP_NAME, clip_vision_encoder_name=CLIP_NAME,
                           load_vision_encoder=False, pretrained=False)
        self.det.clip_text_encoder.model.requires_grad_(False)    # only the prompt tokens train
        self.decoder = MaskDecoder()
        self.aux = nn.Parameter(torch.tensor([10.0, 0.0]))    # score-map read-out: a * cos + b
        self.type_head = nn.Sequential(nn.Linear(3 * 1024, 512), nn.GELU(), nn.Dropout(0.1),
                                       nn.Linear(512, N_TYPES + N_FILTERS))
        self.teacher, self._teacher_text = None, None

    def make_teacher(self):
        """Call after load_m2f2: freeze a copy of the released detector."""
        self.teacher = copy.deepcopy(self.det).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def load_m2f2(self, shard_path):
        """Vision tower + detector from the released LLaVA-M2F2 checkpoint shard."""
        vis, det = {}, {}
        with safe_open(shard_path, framework='pt') as f:
            for k in f.keys():
                if k.startswith('model.vision_tower.vision_tower.'):
                    vis[k[len('model.vision_tower.vision_tower.'):]] = f.get_tensor(k)
                elif k.startswith('deepfake_encoder.'):
                    det[k[len('deepfake_encoder.'):]] = f.get_tensor(k)
        miss_v, _ = self.clip.load_state_dict({k: v.to(self.clip.dtype) for k, v in vis.items()}, strict=False)
        own = self.det.state_dict()
        miss_d, _ = self.det.load_state_dict({k: v.to(own[k].dtype) for k, v in det.items() if k in own}, strict=False)
        assert not [k for k in miss_v if 'position_ids' not in k], miss_v
        assert not miss_d, miss_d[:5]
        return len(vis), len(det)

    def forward(self, pixel):
        with torch.no_grad():
            hs = self.clip(pixel.to(self.clip.dtype), output_hidden_states=True).hidden_states
        hs = [hs[i].float() for i in CLIP_LAYERS]
        out = run_heads(self.det, self.decoder, self.aux, self.type_head, pixel, hs)
        if self.teacher is not None:
            with torch.no_grad():
                t = self.teacher
                if self._teacher_text is None:
                    self._teacher_text = t.text_proj(t.clip_text_encoder().float())
                tv = t.vision_proj(hs[-1])
                ts = F.cosine_similarity(tv[:, 1:], self._teacher_text.unsqueeze(1).expand_as(tv[:, 1:]), dim=-1)
                tdf = t.deepfake_proj(densenet(t, pixel)[3].mean(dim=(2, 3)))
                out['teacher_logits'] = t.output(torch.cat([t.clip_text_alpha * ts, t.clip_vision_alpha * tv[:, 0], tdf], -1))
        return out

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith(('clip.', 'det.clip_text_encoder.model.', 'teacher.'))}
