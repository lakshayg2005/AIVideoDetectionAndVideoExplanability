"""M2F2-Det detector extended with a forgery/processing mask decoder and an edit-type + filter head.

Kept from the paper (weights from the released checkpoint): the LLaVA CLIP ViT-L/14-336 vision tower
(frozen, fine-tuned by M2F2), the forgery-prompt text branch, DenseNet-121 and the real/fake output.
New:
  MaskDecoder - fuses CLIP patch tokens of three layers, the 24x24 forgery-prompt score map and
                DenseNet features (84/42/21 px grids) -> 2 x 84x84 logits: forgery, processing
  aux map     - affine read-out of the paper's 24x24 score map, supervised with the forgery mask
  TypeHead    - [CLS, DenseNet feature, patch features pooled under the predicted mask] ->
                6 forgery types + 6 filters
"""
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

    def densenet(self, x):
        feats = {}
        for name, m in self.det.deepfake_encoder.named_children():
            x = m(x)
            if name in ('denseblock1', 'denseblock2', 'denseblock3'):
                feats[name] = x
        return feats['denseblock1'], feats['denseblock2'], feats['denseblock3'], x

    def patch_contribution(self, scores):
        """The paper's map: exact contribution of each patch score to the fake-real margin."""
        w = self.det.output.weight[1] - self.det.output.weight[0]
        return w[:scores.shape[1]] * self.det.clip_text_alpha * scores

    def forward(self, pixel):
        B = pixel.shape[0]
        with torch.no_grad():
            hs = self.clip(pixel.to(self.clip.dtype), output_hidden_states=True).hidden_states
        hs = [hs[i].float() for i in CLIP_LAYERS]
        v = self.det.vision_proj(hs[-1])
        cls, patches = v[:, 0], v[:, 1:]
        text = self.det.text_proj(self.det.clip_text_encoder().float())
        scores = F.cosine_similarity(patches, text.unsqueeze(1).expand_as(patches), dim=-1)    # [B, 576]
        side = int(scores.shape[1] ** 0.5)
        f1, f2, f3, f4 = self.densenet(pixel)
        dfeat = self.det.deepfake_proj(f4.mean(dim=(2, 3)))
        logits = self.det.output(torch.cat([self.det.clip_text_alpha * scores, self.det.clip_vision_alpha * cls, dfeat], -1))

        score_map = scores.view(B, 1, side, side)
        clip_maps = [h[:, 1:].transpose(1, 2).reshape(B, -1, side, side) for h in hs]
        mask_logits = self.decoder(clip_maps, score_map, f1, f2, f3)
        aux_logits = self.aux[0] * score_map + self.aux[1]

        wgt = torch.sigmoid(F.interpolate(mask_logits[:, :1], size=(side, side), mode='bilinear', align_corners=False))
        wgt = wgt.flatten(1).unsqueeze(-1) + 1e-4
        pooled = (patches * wgt).sum(1) / wgt.sum(1)
        th = self.type_head(torch.cat([cls, dfeat, pooled], -1))
        return {'logits': logits, 'mask_logits': mask_logits, 'aux_logits': aux_logits,
                'type_logits': th[:, :N_TYPES], 'filter_logits': th[:, N_TYPES:],
                'paper_map': self.patch_contribution(scores).view(B, 1, side, side)}

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith('clip.') and not k.startswith('det.clip_text_encoder.model.')}
