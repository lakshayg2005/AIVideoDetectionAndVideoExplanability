"""Face-part masks (DDVQA vocabulary: skin, eyes, eyebrows, nose, mouth) from MediaPipe FaceMesh.

If no landmarks are found (e.g. heavy compression), falls back to a fixed layout for
DDVQA/FF++ style face crops so every image still gets part scores; `used_fallback` reports it.
"""
import cv2
import numpy as np

PARTS = ['skin', 'eyes', 'eyebrows', 'nose', 'mouth']

# FaceMesh indices (468-point topology)
_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
_LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 64, 294, 48, 278, 115, 344, 129, 358, 49, 279, 102, 331]
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
              152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# fallback boxes as fractions (x0, y0, x1, y1) of a roughly centred face crop
_FALLBACK = {
    'eyes': [(0.22, 0.36, 0.45, 0.47), (0.55, 0.36, 0.78, 0.47)],
    'eyebrows': [(0.20, 0.27, 0.46, 0.35), (0.54, 0.27, 0.80, 0.35)],
    'nose': [(0.40, 0.42, 0.60, 0.66)],
    'mouth': [(0.32, 0.68, 0.68, 0.82)],
    'face': [(0.15, 0.20, 0.85, 0.95)],
}

_MESH = None


def _mesh():
    global _MESH
    if _MESH is None:
        import mediapipe as mp
        _MESH = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                 refine_landmarks=False, min_detection_confidence=0.3)
    return _MESH


def _hull(shape, pts, dilate=0):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1)
    if dilate > 0:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m.astype(bool)


def part_masks(img):
    """img: PIL RGB. Returns ({part: bool HxW mask}, used_fallback)."""
    rgb = np.asarray(img.convert('RGB'))
    h, w = rgb.shape[:2]
    res = _mesh().process(rgb)
    if res.multi_face_landmarks:
        lm = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark])
        d = max(3, int(0.03 * min(h, w)))
        m = {
            'eyes': _hull((h, w), lm[_LEFT_EYE], d) | _hull((h, w), lm[_RIGHT_EYE], d),
            'eyebrows': _hull((h, w), lm[_LEFT_BROW], d) | _hull((h, w), lm[_RIGHT_BROW], d),
            'nose': _hull((h, w), lm[_NOSE]),
            'mouth': _hull((h, w), lm[_LIPS], d),
        }
        face = _hull((h, w), lm[_FACE_OVAL])
        used_fallback = False
    else:
        m = {}
        for p, boxes in _FALLBACK.items():
            mask = np.zeros((h, w), bool)
            for x0, y0, x1, y1 in boxes:
                mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
            m[p] = mask
        face = m.pop('face')
        used_fallback = True
    m['skin'] = face & ~(m['eyes'] | m['eyebrows'] | m['nose'] | m['mouth'])
    return {p: m[p] for p in PARTS}, used_fallback


def part_scores(heatmap, masks):
    """Mean heat inside each part (heatmap resized to mask size). Returns {part: score}."""
    h, w = next(iter(masks.values())).shape
    hm = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return {p: float(hm[m].mean()) if m.any() else 0.0 for p, m in masks.items()}


def occlude(img, mask, mode='blur'):
    """Return a copy of img with `mask` region blurred (or mean-filled) for the faithfulness test."""
    a = np.asarray(img.convert('RGB')).copy()
    if mode == 'blur':
        k = max(9, (min(a.shape[:2]) // 8) | 1)
        rep = cv2.GaussianBlur(a, (k, k), 0)
    else:
        rep = np.broadcast_to(a.reshape(-1, 3).mean(0).astype(np.uint8), a.shape)
    a[mask] = rep[mask]
    from PIL import Image
    return Image.fromarray(a)


def overlay(img, heatmap, alpha=0.45):
    """Jet-colour heatmap blended over img (PIL in, PIL out)."""
    from PIL import Image
    a = np.asarray(img.convert('RGB'))
    hm = cv2.resize(heatmap.astype(np.float32), (a.shape[1], a.shape[0]))
    col = cv2.cvtColor(cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray((a * (1 - alpha) + col * alpha).astype(np.uint8))
