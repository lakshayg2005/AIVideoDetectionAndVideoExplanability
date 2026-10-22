"""Stage-A forensic model for the web UI: verdict, forgery type, filters and heatmaps for one image.

Loads M2F2Seg (forensics/model.py) with a Stage-A checkpoint (forensics/train.py). Inputs are prepared as
in training (plain 336 px resize, CLIP normalisation). Runs on the GPU when one is free, else on the CPU
(a few seconds per image), so heatmaps stay available while the GPU trains.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from explain.face_parts import overlay, part_masks
from forensics.data import CLIP_MEAN, CLIP_STD
from forensics.filters import FILTERS
from forensics.model import M2F2Seg

TYPE_NAMES = {
    'real': 'no manipulation',
    'face_swap': 'face swap / reenactment',
    'splice': 'splicing / copy-move (Photoshop)',
    'diffusion_local': 'diffusion inpainting (local edit)',
    'img2img': 'diffusion re-rendering (img2img)',
    'generated': 'fully AI-generated',
}
FILTER_NAMES = {'beauty': 'beauty / skin smoothing', 'blur': 'blur', 'jpeg': 'JPEG compression',
                'color': 'colour grading', 'sharpen': 'sharpening', 'resize': 'down/up-scaling'}


class ForensicEngine:
    def __init__(self, stagea_ckpt, m2f2_shard, device=None):
        from forensics.index import TYPES
        self.types = TYPES
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = M2F2Seg()
        self.model.load_m2f2(m2f2_shard)
        self.model.load_state_dict(torch.load(stagea_ckpt, map_location='cpu')['model'], strict=False)
        self.model.to(self.device).eval()
        if self.device.type == 'cuda':
            self.model.clip.half()

    @torch.no_grad()
    def analyze(self, img, size=336):
        """img: PIL RGB (already face-cropped if wanted). Returns a dict with probabilities and maps."""
        x = np.asarray(img.convert('RGB').resize((size, size), Image.BICUBIC), np.float32) / 255
        pixel = torch.from_numpy(((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
            out = self.model(pixel)
        W, H = img.size
        up = lambda t: F.interpolate(t.float(), size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        masks = torch.sigmoid(out['mask_logits'].float())
        types = torch.softmax(out['type_logits'].float(), -1)[0].cpu().numpy()
        return {
            'p_fake': torch.softmax(out['logits'].float(), -1)[0, 1].item(),
            'types': {self.types[i]: float(p) for i, p in enumerate(types)},
            'filters': {f: float(p) for f, p in zip(FILTERS, torch.sigmoid(out['filter_logits'].float())[0].cpu().numpy())},
            'forgery_map': up(masks[:, :1]),
            'filter_map': up(masks[:, 1:]),
        }

    @staticmethod
    def parts(img, mask, thr=0.3):
        """Face parts (skin, eyes, eyebrows, nose, mouth) whose area is mostly inside the binary mask."""
        pm, fallback = part_masks(img)
        hit = {p: float((mask & m).sum() / m.sum()) for p, m in pm.items() if m.any()}
        return sorted((p for p, c in hit.items() if c > thr), key=lambda p: -hit[p]), fallback


def draw_heatmap(img, prob, color=(255, 40, 40), thr=0.5):
    """Heatmap overlay with the outline of the predicted region (prob > thr)."""
    out = np.asarray(overlay(img, prob, alpha=0.4)).copy()
    contours, _ = cv2.findContours((prob > thr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, max(2, min(img.size) // 150))
    return Image.fromarray(out)


def region_text(prob, thr=0.5):
    m = prob > thr
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() / m.shape[0], xs.mean() / m.shape[1]
    v = 'upper' if cy < 0.33 else 'lower' if cy > 0.66 else 'middle'
    h = 'left' if cx < 0.33 else 'right' if cx > 0.66 else 'centre'
    return f'{v} {h}'.replace('middle centre', 'centre'), 100 * m.mean()


def smart_crop(img, crop_face, min_frac=0.5):
    """Crop the largest face unless the image already is a tight face crop (face box >= min_frac of the
    width), so inputs keep the framing seen in training. Returns (face, box, is_face)."""
    face, box = crop_face(img)
    if box is None:    # Haar misses small / blurry faces: accept the image as a face crop if MediaPipe finds one
        return img, None, not part_masks(img)[1]
    if (box[2] - box[0]) >= min_frac * img.size[0]:
        return img, None, True
    return face, box, True
