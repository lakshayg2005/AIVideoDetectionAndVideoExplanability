"""Inference engine for the M2F2-Det web UI.

Wraps the same loading / prompting path used by llava/serve/cli_DDVQA_det.py and
cli_DDVQA_exp.py, and adds what a real-world upload needs: face cropping, video
frame sampling, a fake-probability score, and per-video aggregation.

`MockEngine` mimics the interface without torch/weights so the UI can be tried on
a machine without a GPU (python -m app.webui --mock).
"""
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

SYSTEM = ("Assistant: A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")
DET_PROMPT = f"{SYSTEM}###Human: <image>\n <deepfake>\n Determine the authenticity. Is the image real or fake? ###Assistant:"
EXP_PROMPT = f"{SYSTEM}###Human: <image>\n<deepfake>\n{{question}} ###Assistant:"

PARTS = ["skin", "eyes", "nose", "mouth", "eyebrows"]

# Question templates copied from utils/DDVQA_eval/c40/test.jsonl (what the model was tuned on).
FAKE_EXAMPLES = {
    "skin": "The skin is overly smooth and lacks of details, such as skin texture. ",
    "eyes": "The eyes look unnaturally asymmetric. ",
    "nose": "The nose lacks of subtle details, such as pores, fine lines. ",
    "mouth": "The mouth is misaligned with other facial features. ",
    "eyebrows": "The eyebrows look asymmetric. ",
}


def _verb(part):
    return "are" if part.endswith("s") else "is"


def build_questions(label, parts):
    """Return [(title, question)] for the explanation step, conditioned on the verdict."""
    qs = []
    if parts:
        qs.append(("Overall", f"This image is {label}. Describe the authenticity of the image based on {', '.join(parts)}."))
    for p in parts:
        Cap = p.capitalize()
        if label == "fake":
            if p == "eyes":
                q = (f"The eyes in the image are fake, tell me why. Answer in one sentence in the format\"The eyes ……\" like. "
                     f"Instead of saying, \"The eyes are fake because\" directly answer why \"{FAKE_EXAMPLES[p]}\"")
            else:
                q = (f"The {p} in the image {_verb(p)} fake, tell me why. Answer in one sentence in the format\"The {p} ……\". "
                     f"Instead of saying, \"The {p} {_verb(p)} fake because\" directly answer why like \"{FAKE_EXAMPLES[p]}\"")
        else:
            q = (f"The {p} in the image {_verb(p)} real, tell me why. Answer in one sentence in the format"
                 f"\"The person has naturally …… {p}.\"")
        qs.append((Cap, q))
    return qs


# --------------------------------------------------------------------------- faces

_FACE_CASCADE = None


def _cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    return _FACE_CASCADE


def crop_face(img, margin=0.35):
    """Crop the largest face with a margin (DDVQA/FF++ crops include some context).

    Returns (crop, box) where box is (x0, y0, x1, y1) in the original image, or
    (img, None) if no face was found.
    """
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    min_side = max(40, min(gray.shape) // 10)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    if len(faces) == 0:
        return img, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * (1 + 2 * margin)
    W, H = img.size
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return img.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def draw_box(img, box, label=None, p_fake=None):
    out = img.copy()
    if box is None:
        return out
    color = (220, 50, 47) if label == "fake" else (40, 167, 69)
    d = ImageDraw.Draw(out)
    width = max(2, min(img.size) // 150)
    d.rectangle(box, outline=color, width=width)
    if label is not None:
        text = f"{label.upper()} {p_fake:.0%}" if p_fake is not None else label.upper()
        tw = d.textlength(text)
        d.rectangle((box[0], max(0, box[1] - 14), box[0] + tw + 8, box[1]), fill=color)
        d.text((box[0] + 4, max(0, box[1] - 13)), text, fill=(255, 255, 255))
    return out


# --------------------------------------------------------------------------- video

def sample_frames(video_path, n):
    """Evenly sample up to n RGB frames. Returns [(timestamp_sec, PIL.Image)]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, num=min(n, total)).round().astype(int)
        for i in sorted(set(idxs.tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames.append((i / fps, Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
    if not frames:  # some containers report no frame count; fall back to reading sequentially
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        allf, i = [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append((i / fps, f))
            i += 1
        for j in np.linspace(0, len(allf) - 1, num=min(n, len(allf))).round().astype(int) if allf else []:
            t, f = allf[j]
            frames.append((t, Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
    cap.release()
    return frames


def aggregate(p_fakes, trim=0.1):
    """Trimmed mean of frame scores (the paper's video protocol drops extreme frames)."""
    s = sorted(p_fakes)
    k = int(len(s) * trim)
    if len(s) - 2 * k >= 1:
        s = s[k:len(s) - k]
    return float(np.mean(s))


# --------------------------------------------------------------------------- engines

class M2F2Engine:
    def __init__(self, model_path, load_4bit=True, load_8bit=False):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX
        from llava.mm_utils import get_model_name_from_path, tokenizer_hybrid_token
        from llava.model.builder import load_deepfake_model

        self.torch = torch
        self._tok_fn = lambda p: tokenizer_hybrid_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX,
                                                        return_tensors="pt").unsqueeze(0).to(self.model.device)
        name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, _, _ = load_deepfake_model(model_path, None, name, load_8bit, load_4bit, device="cuda")
        self.model.eval()
        if not (load_4bit or load_8bit):
            self.model = self.model.to("cuda")
        self.model.load_deepfake_encoder(self.model.config.deepfake_model_path, verbose=False)
        self.model.deepfake_encoder.to("cuda", dtype=torch.float16)

        # First token where "The image is real" and "The image is fake" differ -> the decision token.
        r = self.tokenizer.encode("The image is real", add_special_tokens=False)
        f = self.tokenizer.encode("The image is fake", add_special_tokens=False)
        i = next(k for k in range(min(len(r), len(f))) if r[k] != f[k])
        self.real_id, self.fake_id = r[i], f[i]

    def _generate(self, prompt, img, max_new_tokens):
        with self.torch.inference_mode():
            out = self.model.generate(
                self._tok_fn(prompt), images=[img], image_sizes=[img.size], deepfake_inputs=[img],
                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                output_scores=True, return_dict_in_generate=True)
        return out

    def detect(self, img):
        """Returns (label, p_fake, raw_text)."""
        out = self._generate(DET_PROMPT, img, 20)
        scores = out["scores"]
        gen = out["sequences"][0][-len(scores):].tolist()
        text = self.tokenizer.decode(out["sequences"][0], skip_special_tokens=True).strip()
        p_fake = None
        for step, tok in enumerate(gen):
            if tok in (self.real_id, self.fake_id):
                logits = scores[step][0, [self.real_id, self.fake_id]].float()
                p_fake = self.torch.softmax(logits, dim=0)[1].item()
                break
        if p_fake is None:  # model phrased it differently; fall back to the text
            p_fake = 1.0 if "fake" in text.lower() else 0.0
        return ("fake" if p_fake >= 0.5 else "real"), p_fake, text

    def ask(self, img, question, max_new_tokens=160):
        out = self._generate(EXP_PROMPT.format(question=question), img, max_new_tokens)
        return self.tokenizer.decode(out["sequences"][0], skip_special_tokens=True).strip()


class MockEngine:
    """Random but deterministic-per-image outputs, for trying the UI without the model."""

    def detect(self, img):
        rnd = random.Random(hash(np.asarray(img.resize((16, 16))).tobytes()))
        p = rnd.random()
        label = "fake" if p >= 0.5 else "real"
        return label, p, f"[MOCK] The image is {label}."

    def ask(self, img, question, max_new_tokens=160):
        return "[MOCK] " + question.split(".")[0] + " — this is placeholder text; run without --mock for real output."
