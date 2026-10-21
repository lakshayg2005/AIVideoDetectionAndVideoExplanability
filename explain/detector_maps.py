"""Spatial evidence maps for the M2F2-Det binary detector.

The detector head (llava/model/deepfake/M2F2Det/model.py) computes
    logits = W @ [alpha_t * clip_scores (576) | alpha_v * clip_cls (1024) | densenet_feat (1024)] + b
so each of the 576 CLIP patch scores enters the fake-vs-real margin *linearly*. That gives an
exact per-patch attribution (`patch_map`) with no approximation. The DenseNet branch goes through
avgpool + Linear + LayerNorm, so for it we use Grad-CAM on the last DenseNet feature map
(`gradcam_map`). Both are returned on the 336x336 CLIP input grid together with branch totals,
which say how much of the decision came from each branch.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module

FAKE_IDX, REAL_IDX = 1, 0


def _norm01(m):
    m = np.maximum(m, 0)
    return m / m.max() if m.max() > 0 else m


class DetectorExplainer:
    def __init__(self, llava_model, device='cuda'):
        det = llava_model.deepfake_encoder
        self.device = device
        self.vision_tower = llava_model.get_vision_tower()
        self.select_layer = llava_model.config.mm_vision_select_layer
        self.processor = llava_model.processors['deepfake_processor']
        # fp32 copies of the small heads so gradients are stable; the big parts stay as loaded.
        # Copies drop accelerate's device_map hooks, which would otherwise keep them in fp16.
        def fp32(m):
            m = copy.deepcopy(m)
            remove_hook_from_module(m, recurse=True)
            return m.float().to(device).eval()
        self.densenet = fp32(det.deepfake_encoder)
        self.deepfake_proj = fp32(det.deepfake_proj)
        self.vision_proj = fp32(det.vision_proj)
        self.output = fp32(det.output)
        self.alpha_v = det.clip_vision_alpha.detach().float().to(device)
        self.alpha_t = det.clip_text_alpha.detach().float().to(device)
        with torch.no_grad():
            text = det.clip_text_encoder().to(det.text_proj[0].weight.dtype)
            self.text_feat = det.text_proj(text).float().to(device)    # [1, 1024]

    def preprocess(self, img):
        return self.processor.preprocess(img, return_tensors='pt')['pixel_values'].to(self.device)

    @torch.no_grad()
    def clip_features(self, pixel):
        out = self.vision_tower.vision_tower(pixel.to(self.vision_tower.dtype), output_hidden_states=True)
        return out.hidden_states[self.select_layer].float()    # [B, 577, 1024]

    def explain(self, img, pixel=None):
        """Returns dict with p_fake, maps on a 336x336 grid in [0, 1], and branch contributions."""
        pixel = self.preprocess(img) if pixel is None else pixel
        clip_feat = self.clip_features(pixel)
        v = self.vision_proj(clip_feat)
        cls, patches = v[:, 0], v[:, 1:]
        scores = F.cosine_similarity(patches, self.text_feat.unsqueeze(1).expand_as(patches), dim=-1)    # [1, 576]

        with torch.enable_grad():
            fmap = self.densenet(pixel.float().requires_grad_(True))    # [1, 1024, h, w]; weights stay frozen
            fmap.retain_grad()
            dfeat = self.deepfake_proj(fmap.mean(dim=(2, 3)))
            feats = torch.cat([self.alpha_t * scores, self.alpha_v * cls, dfeat], dim=-1)
            logits = self.output(feats)
            margin = logits[0, FAKE_IDX] - logits[0, REAL_IDX]
            margin.backward()

        w = (self.output.weight[FAKE_IDX] - self.output.weight[REAL_IDX]).detach()
        n = scores.shape[1]
        patch_contrib = (w[:n] * self.alpha_t * scores[0]).detach()
        side = int(round(n ** 0.5))
        patch_map = patch_contrib.view(1, 1, side, side)

        cam = F.relu((fmap.grad.mean(dim=(2, 3), keepdim=True) * fmap).sum(1, keepdim=True)).detach()

        H, W = pixel.shape[-2:]
        up = lambda m: F.interpolate(m, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        patch_np, cam_np = _norm01(up(patch_map)), _norm01(up(cam))
        return {
            'p_fake': torch.softmax(logits.detach(), -1)[0, FAKE_IDX].item(),
            'margin': margin.item(),
            'patch_map': patch_np,
            'gradcam_map': cam_np,
            'fused_map': _norm01(0.5 * patch_np + 0.5 * cam_np),
            'branch': {
                'clip_patches': patch_contrib.sum().item(),
                'clip_cls': (w[n:n + cls.shape[1]] @ (self.alpha_v * cls[0])).item(),
                'densenet': (w[n + cls.shape[1]:] @ dfeat[0]).item(),
                'bias': (self.output.bias[FAKE_IDX] - self.output.bias[REAL_IDX]).item(),
            },
        }

    def p_fake(self, img):
        return self.score(img)[0]

    @torch.no_grad()
    def score(self, img):
        """Detector-only (p_fake, fake-real logit margin); the margin does not saturate like p_fake,
        so it is what the occlusion / faithfulness test compares."""
        pixel = self.preprocess(img)
        v = self.vision_proj(self.clip_features(pixel))
        cls, patches = v[:, 0], v[:, 1:]
        scores = F.cosine_similarity(patches, self.text_feat.unsqueeze(1).expand_as(patches), dim=-1)
        dfeat = self.deepfake_proj(self.densenet(pixel.float()).mean(dim=(2, 3)))
        logits = self.output(torch.cat([self.alpha_t * scores, self.alpha_v * cls, dfeat], dim=-1))
        return torch.softmax(logits, -1)[0, FAKE_IDX].item(), (logits[0, FAKE_IDX] - logits[0, REAL_IDX]).item()
