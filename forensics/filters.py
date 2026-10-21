"""Photo filters ("heavy edits") with exact labels, shared by the offline generator
(scripts/make_filtered_data.py) and on-the-fly augmentation in Stage-A training.

apply() returns the filtered image, the edit mask (skin only for the local beauty retouch, else the
whole image) and [{name, strength}] in the order applied.
"""
import cv2
import numpy as np
from PIL import Image

DESCRIBE = {
    'beauty': 'a beauty filter has smoothed the skin ({s})',
    'blur': 'the whole image is blurred ({s})',
    'jpeg': 'it has been saved with {s} JPEG compression, leaving blocky artifacts',
    'color': 'the colours have been graded ({s} change in contrast, saturation and tone)',
    'sharpen': 'it has been sharpened ({s}), adding halos around edges',
    'resize': 'it has been downscaled and upscaled again ({s}), losing fine detail',
}
FILTERS = list(DESCRIBE)


def level(x):
    return 'strong' if x > 0.66 else 'moderate' if x > 0.33 else 'light'


def beauty(a, skin, t):
    d = int(5 + 10 * t)
    smooth = cv2.bilateralFilter(a, d, 30 + 70 * t, 30 + 70 * t)
    m = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 3)[..., None]
    return (a * (1 - m) + smooth * m).astype(np.uint8)


def blur(a, t):
    return cv2.GaussianBlur(a, (0, 0), 0.8 + 3.2 * t)


def jpeg(a, t):
    q = int(50 - 42 * t)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def color(a, t, rng):
    x = a.astype(np.float32) / 255
    x = (x - 0.5) * (1 + rng.uniform(-0.6, 0.6) * t) + 0.5                      # contrast
    g = x.mean(-1, keepdims=True)
    x = g + (x - g) * (1 + rng.uniform(-0.8, 0.8) * t)                          # saturation
    x = x * (1 + np.array([rng.uniform(-.25, .25), 0, rng.uniform(-.25, .25)]) * t)   # warm / cool tint
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def sharpen(a, t):
    b = cv2.GaussianBlur(a, (0, 0), 2)
    return cv2.addWeighted(a, 1 + 2 * t, b, -2 * t, 0)


def resize(a, t):
    h, w = a.shape[:2]
    f = 1 - 0.75 * t
    small = cv2.resize(a, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply(img, rng, n_max=2, allow=None, skin=None):
    """img: PIL RGB; allow: filter names that may be used (e.g. no beauty for non-face images);
    skin: optional precomputed bool skin mask (otherwise MediaPipe is run when beauty is drawn)."""
    a = np.asarray(img.convert('RGB')).copy()
    names = rng.sample(list(allow or FILTERS), rng.randint(1, n_max))
    if 'jpeg' in names:    # compression is normally the last step
        names = [n for n in names if n != 'jpeg'] + ['jpeg']
    edit = np.zeros(a.shape[:2], bool)
    applied = []
    for n in names:
        t = rng.uniform(0.2, 1.0)
        if n == 'beauty':
            if skin is None:
                from explain.face_parts import part_masks
                skin = part_masks(img)[0]['skin']
            if skin.sum() < 0.05 * skin.size:
                continue
            a, edit = beauty(a, skin, t), edit | skin
        else:
            a = {'blur': blur, 'jpeg': jpeg, 'sharpen': sharpen, 'resize': resize}[n](a, t) if n != 'color' else color(a, t, rng)
            edit[:] = True
        applied.append({'name': n, 'strength': round(t, 3)})
    return Image.fromarray(a), edit, applied


def explanation(label, applied):
    edits = '; '.join(DESCRIBE[f['name']].format(s=level(f['strength'])) for f in applied)
    if label == 'real':
        return (f'This image is real. The face itself has not been manipulated, but the photo has been edited: {edits}. '
                f'These are photo filters, not a deepfake, so they should not be mistaken for forgery traces.')
    return (f'This image is fake. The face has been manipulated, and the photo has also been edited: {edits}. '
            f'The filters can hide some forgery traces, so the decision relies on the remaining facial inconsistencies.')
