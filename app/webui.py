"""M2F2-Det web UI: upload an image or a video, get a real/fake verdict plus explanations.

Run from the repo root:
    python -m app.webui                       # real model (needs the checkpoints + a CUDA GPU)
    python -m app.webui --mock                # UI only, fake outputs, no GPU needed
"""
import argparse
import time

import gradio as gr
import pandas as pd

from app.engine import PARTS, MockEngine, aggregate, build_questions, crop_face, draw_box, sample_frames

ENGINE = None
MOCK = False


def _verdict_md(label, p_fake, extra=""):
    color = "#dc322f" if label == "fake" else "#28a745"
    word = "LIKELY FAKE (manipulated)" if label == "fake" else "LIKELY REAL"
    banner = ("\n\n> **Demo mode** — outputs are random placeholders, not model predictions."
              if MOCK else "")
    return (f"<h2 style='color:{color};margin:0'>{word}</h2>"
            f"Fake probability: **{p_fake:.1%}** {extra}{banner}")


def _explain(img, label, parts):
    if not parts:
        return "_Explanations turned off._"
    lines = []
    for title, q in build_questions(label, parts):
        lines.append(f"**{title}:** {ENGINE.ask(img, q)}")
    return "\n\n".join(lines)


def analyze_image(image, use_crop, parts, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image.")
    t0 = time.time()
    image = image.convert("RGB")
    face, box = crop_face(image) if use_crop else (image, None)
    progress(0.2, desc="Detecting")
    label, p_fake, raw = ENGINE.detect(face)
    progress(0.5, desc="Generating explanations")
    expl = _explain(face, label, parts)
    note = "" if (box or not use_crop) else "<br/>_No face detected — analysed the whole image._"
    return (
        _verdict_md(label, p_fake, note),
        {"Fake": p_fake, "Real": 1 - p_fake},
        draw_box(image, box, label, p_fake) if box else image,
        face,
        expl,
        f"Model answer: `{raw}`  ·  {time.time() - t0:.1f}s",
    )


def analyze_video(video, n_frames, use_crop, parts, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a video.")
    t0 = time.time()
    frames = sample_frames(video, int(n_frames))
    if not frames:
        raise gr.Error("Could not read any frames from this video.")
    rows, gallery, faces = [], [], []
    no_face = 0
    for i, (ts, fr) in enumerate(progress.tqdm(frames, desc="Analysing frames")):
        face, box = crop_face(fr) if use_crop else (fr, None)
        if use_crop and box is None:
            no_face += 1
        label, p, _ = ENGINE.detect(face)
        rows.append({"time (s)": round(ts, 2), "fake probability": p})
        faces.append((p, face))
        gallery.append((draw_box(fr, box, label, p) if box else fr, f"t={ts:.1f}s · {label} {p:.0%}"))

    p_video = aggregate([r["fake probability"] for r in rows])
    label = "fake" if p_video >= 0.5 else "real"
    n_fake = sum(r["fake probability"] >= 0.5 for r in rows)

    # Explain using the frame that most strongly supports the verdict.
    key_p, key_face = (max if label == "fake" else min)(faces, key=lambda x: x[0])
    progress(0.9, desc="Generating explanations")
    expl = _explain(key_face, label, parts)

    extra = f"<br/>{n_fake} of {len(rows)} sampled frames flagged fake (trimmed-mean video score)."
    if no_face:
        extra += f"<br/>_No face found in {no_face} frame(s); whole frame used there._"
    df = pd.DataFrame(rows)
    return (
        _verdict_md(label, p_video, extra),
        {"Fake": p_video, "Real": 1 - p_video},
        gr.LinePlot(value=df, x="time (s)", y="fake probability", y_lim=[0, 1], title="Per-frame fake probability",
                    tooltip=["time (s)", "fake probability"], height=260),
        gallery,
        key_face,
        expl,
        f"{len(rows)} frames · {time.time() - t0:.1f}s",
    )


def build_ui():
    with gr.Blocks(title="M2F2-Det Deepfake Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# M2F2-Det Deepfake Analyzer\n"
                    "Upload a face image or video. The model decides **real vs. fake** and explains *why*, "
                    "facial part by part. Works best on clear, front-facing faces.")
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_in = gr.Image(type="pil", label="Image", height=320)
                        img_crop = gr.Checkbox(value=True, label="Auto-crop the largest face")
                        img_parts = gr.CheckboxGroup(PARTS, value=PARTS, label="Explain these facial parts")
                        img_btn = gr.Button("Analyze image", variant="primary")
                    with gr.Column(scale=1):
                        img_verdict = gr.Markdown()
                        img_label = gr.Label(label="Confidence", num_top_classes=2)
                        with gr.Row():
                            img_annot = gr.Image(label="Detected face", height=220)
                            img_face = gr.Image(label="Model input", height=220)
                        img_expl = gr.Markdown(label="Explanation")
                        img_meta = gr.Markdown()
                img_btn.click(analyze_image, [img_in, img_crop, img_parts],
                              [img_verdict, img_label, img_annot, img_face, img_expl, img_meta])

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label="Video", height=320)
                        vid_n = gr.Slider(4, 48, value=16, step=1, label="Frames to sample",
                                          info="More frames = more reliable but slower (~1–2 s per frame on an RTX 3060).")
                        vid_crop = gr.Checkbox(value=True, label="Auto-crop the largest face")
                        vid_parts = gr.CheckboxGroup(PARTS, value=PARTS, label="Explain these facial parts")
                        vid_btn = gr.Button("Analyze video", variant="primary")
                    with gr.Column(scale=1):
                        vid_verdict = gr.Markdown()
                        vid_label = gr.Label(label="Confidence", num_top_classes=2)
                        vid_plot = gr.LinePlot()
                        vid_key = gr.Image(label="Key frame used for the explanation", height=220)
                        vid_expl = gr.Markdown(label="Explanation")
                        vid_meta = gr.Markdown()
                vid_gallery = gr.Gallery(label="Sampled frames", columns=6, height="auto")
                vid_btn.click(analyze_video, [vid_in, vid_n, vid_crop, vid_parts],
                              [vid_verdict, vid_label, vid_plot, vid_gallery, vid_key, vid_expl, vid_meta])

        gr.Markdown("<small>M2F2-Det (CVPR 2025) — LLaVA-1.5-7B + CLIP + DenseNet-121, 4-bit. "
                    "Scores are model confidence, not proof; use as an aid, not a final judgment.</small>")
    return demo


def main():
    global ENGINE, MOCK
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="./checkpoints/llava-v1.5-7b-M2F2-Det")
    ap.add_argument("--mock", action="store_true", help="UI demo without loading the model")
    ap.add_argument("--no-4bit", action="store_true", help="load in fp16 (needs ~18 GB VRAM)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = ap.parse_args()

    MOCK = args.mock
    if MOCK:
        ENGINE = MockEngine()
    else:
        from app.engine import M2F2Engine
        print(f"Loading M2F2-Det from {args.model_path} ...")
        ENGINE = M2F2Engine(args.model_path, load_4bit=not args.no_4bit)
        print("Model ready.")
    build_ui().queue(max_size=8).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
