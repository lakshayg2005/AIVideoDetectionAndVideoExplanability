"""Forensic web UI: verdict, forgery type, filters, heatmaps and a heatmap-grounded explanation.

The Stage-A model (app/forensic_engine.py) gives the verdict, type, filters and the forgery / filter heatmaps.
The explanation is written by the M2F2-Det LLM about the face parts under the predicted forged region
(explain/grounded_explain.py idea), so text and heatmap refer to the same evidence.

    python -m app.forensic_webui                    # heatmaps + LLM explanations (GPU)
    python -m app.forensic_webui --no-llm           # heatmaps only, works on the CPU
"""
import argparse
import time

import gradio as gr
import pandas as pd

from app.engine import PARTS, aggregate, build_questions, crop_face, draw_box, sample_frames
from app.forensic_engine import (FILTER_NAMES, TYPE_NAMES, ForensicEngine, draw_heatmap, position_consistent,
                                 region_text, smart_crop)

FORENSIC, LLM = None, None
SELF_EXPLAIN = False    # Stage-C forensic LLM: one open question, verdict and explanation from its own answer
OPEN_QUESTION = 'Is this image real or fake? Explain why and point out any manipulated region.'


def _join(xs):
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


def _pct(x):
    return '<1%' if x < 1 else f'{x:.0f}%'


def _evidence(r, face, is_face, label=None):
    """Deterministic, heatmap-grounded summary + the parts under the predicted forged region."""
    label = label or ('fake' if r['p_fake'] >= 0.5 else 'real')
    fake_types = {k: v for k, v in r['types'].items() if k != 'real'}
    t = max(fake_types, key=fake_types.get) if label == 'fake' else max(r['types'], key=r['types'].get)    # name a forgery type for fakes
    filters = [f for f, p in sorted(r['filters'].items(), key=lambda x: -x[1]) if p >= 0.5]
    lines, parts = [], []
    if label == 'fake':
        conf = f"{r['types'][t]:.0%}" if r['types'][t] >= 0.3 else 'low confidence'
        lines.append(f"**Manipulation type:** {TYPE_NAMES.get(t, t)} ({conf})")
        reg = region_text(r['forgery_map'])
        if reg and is_face:
            parts, fallback = FORENSIC.parts(face, r['forgery_map'] > 0.5)
            if fallback:    # no landmarks: do not name face parts from the fixed layout
                parts = []
            where = _join(parts) if parts else f'the {reg[0]} of the face'
            lines.append(f"**Forged region (red outline):** {where}, {_pct(reg[1])} of the face crop"
                         + (' _(face landmarks not found, approximate parts)_' if fallback else ''))
        elif reg:
            lines.append(f"**Forged region (red outline):** {reg[0]} of the image, {_pct(reg[1])} of it")
        else:
            lines.append('**Forged region:** spread over the whole image (no single edited area, typical of fully generated or re-rendered faces)')
    elif r['p_fake'] >= 0.5 and region_text(r['forgery_map']):
        lines.append('**The forensic model still marks a region (red outline)**, but the final verdict is real — see the warning above.')
    else:
        lines.append('**No manipulated region found.**')
    if filters:
        lines.append('**Photo filters detected (blue outline):** ' + ', '.join(f'{FILTER_NAMES[f]} ({r["filters"][f]:.0%})' for f in filters)
                     + ' — edits, not the forgery itself')
    return label, t, parts, '\n\n'.join('- ' + l for l in lines)


def _explanation(face, label, parts, is_face=True):
    if LLM is None:
        return '_Language-model explanation is off (`--no-llm`, e.g. while the GPU is training)._'
    if not is_face:
        return ('_The released M2F2-Det language model is trained on faces only, so no text is generated for this '
                'image; the evidence above comes from the forensic model._')
    if label == 'fake' and not parts:
        return f"**Overall:** {LLM.ask(face, 'Is this face real or fake? Explain why.')}"
    qs = build_questions(label, parts if label == 'fake' else PARTS[:3])
    return '\n\n'.join(f'**{title}:** {LLM.ask(face, q)}' for title, q in qs)


def _verdict_md(label, p_fake, t, extra='', uncertain=False):
    if uncertain:
        color, word = '#d08700', f"UNCERTAIN — leaning {label}" + (f' ({TYPE_NAMES.get(t, t)})' if label == 'fake' else '')
    else:
        color = '#dc322f' if label == 'fake' else '#28a745'
        word = f'LIKELY FAKE — {TYPE_NAMES.get(t, t)}' if label == 'fake' else 'LIKELY REAL'
    return f"<h2 style='color:{color};margin:0'>{word}</h2>Fake probability: **{p_fake:.1%}** {extra}"


def _signals(r, explainer_label=None):
    """Agreement of the detector, the type head and (if run) the explainer. Uncertain unless all agree."""
    det = 'fake' if r['p_fake'] >= 0.5 else 'real'
    typ = 'real' if r['types']['real'] >= 0.5 else 'fake'
    votes = [('detector', det), ('type head', typ)] + ([('explainer', explainer_label)] if explainer_label else [])
    agree = all(v == det for _, v in votes)
    txt = ' · '.join(f'{n}: {v}' for n, v in votes)
    return det, not agree, f'<br/>Signals — {txt}' + ('' if agree else '<br/>⚠️ _The signals disagree: treat this result as uncertain._')


AREA = __import__('re').compile(r'(\d+(?:\.\d+)?)\s*%')
LIGHTING = __import__('re').compile(r'\b(lighting|light|shadows?|illuminat\w*)\b', __import__('re').I)


def _text_checks(text, r, region):
    """Warnings where the text claims more than the heatmap shows."""
    warn = []
    reg = region_text(r['forgery_map'])
    said = position_consistent(text, region)
    if reg is None and (AREA.search(text) or said is not None):
        warn.append('the text names a location / size, but the heatmap does not localise a clear region')
    elif said is False:
        warn.append(f'the text describes a different area than the heatmap ({region}); trust the heatmap for the location')
    if reg is not None:
        nums = [float(x) for x in AREA.findall(text)]
        if nums and all(not (0.5 * reg[1] <= n <= 2 * reg[1] + 1) for n in nums):
            warn.append(f'the size stated in the text does not match the heatmap ({reg[1]:.0f}%)')
    if LIGHTING.search(text):
        warn.append('lighting / shadow statements are the model\'s description, not a measurement')
    return ''.join(f'\n\n⚠️ _{w}._' for w in warn)


def _text_verdict(text):
    from explain.eval_selfexplain import verdict
    return verdict(text)


def _grounding(r, is_face, label):
    """Where the heatmap points, as a sentence for the prompt (None if there is no marked region)."""
    reg = region_text(r['forgery_map'])
    if label != 'fake':
        return None, None
    if not reg:
        return ('The forensic heatmap does not localise a clear manipulated region; '
                'do not guess a location if the evidence is not localised.'), None
    if is_face:
        parts, fallback = FORENSIC.parts(FORENSIC_LAST_FACE[0], r['forgery_map'] > 0.5)
        if parts and not fallback:
            return f"The forensic heatmap marks the {', '.join(parts)} ({reg[1]:.0f}% of the face).", reg[0]
    return f"The forensic heatmap marks the {reg[0]} of the image ({reg[1]:.0f}% of it).", reg[0]


FORENSIC_LAST_FACE = [None]


def _self_explain(face, r, is_face=True):
    """Stage-C model. The verdict is the forensic heads' (more accurate than the explainer's own on every test set
    except compressed FF++, RESULTS.md section 5); the explainer answers the open question, and when it disagrees it
    is also asked to explain the heads' verdict."""
    FORENSIC_LAST_FACE[0] = face
    det_label = 'fake' if r['p_fake'] >= 0.5 else 'real'
    hint, region = _grounding(r, is_face, det_label)
    question = f'{hint} {OPEN_QUESTION}' if hint else OPEN_QUESTION    # explain where the heatmap points
    text = LLM.ask(face, question, max_new_tokens=220)
    own = _text_verdict(text) or det_label
    _, uncertain, note = _signals(r, own)
    if own != det_label:
        guided = LLM.ask(face, f'This image is {det_label}. {hint or ""} Explain why and point out any manipulated region.',
                         max_new_tokens=220)
        text = f'**Explanation of the forensic verdict ({det_label}):** {guided}\n\n**Explainer\'s own answer:** {text}'
        check_text = guided
    else:
        check_text, text = text, f'**Explanation:** {text}'
    text += _text_checks(check_text, r, region)
    return det_label, r['p_fake'], note, text, uncertain


def _verdicts(face, r, is_face, use_llm):
    """Final verdict: the language model's on faces when it runs (better calibrated on compressed FF++ faces),
    else the forensic model's; disagreement is reported."""
    p_det = r['p_fake']
    if LLM is None or not use_llm or not is_face:
        return ('fake' if p_det >= 0.5 else 'real'), p_det, ''
    label, p_llm, _ = LLM.detect(face)
    det_label = 'fake' if p_det >= 0.5 else 'real'
    note = f'<br/>Forensic model: {det_label} ({p_det:.0%} fake) · language model: {label} ({p_llm:.0%} fake)'
    if det_label != label:
        note += '<br/>⚠️ _The two models disagree: treat this result as uncertain._'
    return label, p_llm, note


def analyze_image(image, use_crop, use_llm, progress=gr.Progress()):
    if image is None:
        raise gr.Error('Please upload an image.')
    t0 = time.time()
    image = image.convert('RGB')
    face, box, is_face = smart_crop(image, crop_face) if use_crop else (image, None, False)
    progress(0.2, desc='Forensic analysis')
    r = FORENSIC.analyze(face)
    if SELF_EXPLAIN and use_llm and LLM is not None:
        progress(0.5, desc='Writing the explanation')
        label, p_final, vnote, expl, uncertain = _self_explain(face, r, is_face)
        label, t, parts, ev = _evidence(r, face, is_face, label)
    else:
        label, p_final, vnote = _verdicts(face, r, is_face, use_llm)
        uncertain = _signals(r)[1] or 'disagree' in vnote
        label, t, parts, ev = _evidence(r, face, is_face, label)
        progress(0.5, desc='Writing the explanation')
        expl = _explanation(face, label, parts, is_face) if use_llm else '_Explanation not requested._'
    note = vnote + ('' if (is_face or not use_crop) else '<br/>_No face detected — analysed the whole image._')
    types = {TYPE_NAMES[k]: v for k, v in sorted(r['types'].items(), key=lambda x: -x[1])[:3]}
    return (_verdict_md(label, p_final, t, note, uncertain), types,
            draw_heatmap(face, r['forgery_map']), draw_heatmap(face, r['filter_map'], color=(40, 120, 255)),
            draw_box(image, box, label, p_final) if box else None, ev, expl, f'{time.time() - t0:.1f}s')


def analyze_video(video, n_frames, use_crop, use_llm, progress=gr.Progress()):
    if not video:
        raise gr.Error('Please upload a video.')
    t0 = time.time()
    frames = sample_frames(video, int(n_frames))
    if not frames:
        raise gr.Error('Could not read any frames from this video.')
    rows, gallery, faces = [], [], []
    for ts, fr in progress.tqdm(frames, desc='Analysing frames'):
        face, box, is_face = smart_crop(fr, crop_face) if use_crop else (fr, None, False)
        r = FORENSIC.analyze(face)
        rows.append({'time (s)': round(ts, 2), 'fake probability': r['p_fake']})
        faces.append((r['p_fake'], face, r, is_face))
        gallery.append((draw_heatmap(face, r['forgery_map']), f"t={ts:.1f}s · fake {r['p_fake']:.0%}"))
    p_video = aggregate([x['fake probability'] for x in rows])
    label = 'fake' if p_video >= 0.5 else 'real'
    key_p, key_face, key_r, key_is_face = (max if label == 'fake' else min)(faces, key=lambda x: x[0])
    _, t, parts, ev = _evidence(key_r, key_face, key_is_face)
    if SELF_EXPLAIN and use_llm and LLM is not None:
        expl = _self_explain(key_face, key_r, key_is_face)[3]
    else:
        expl = _explanation(key_face, label, parts, key_is_face) if use_llm else '_Explanation not requested._'
    n_fake = sum(x['fake probability'] >= 0.5 for x in rows)
    df = pd.DataFrame(rows)
    return (_verdict_md(label, p_video, t, f'<br/>{n_fake} of {len(rows)} sampled frames flagged fake.'),
            gr.LinePlot(value=df, x='time (s)', y='fake probability', y_lim=[0, 1], title='Per-frame fake probability', height=240),
            gallery, draw_heatmap(key_face, key_r['forgery_map']), ev, expl, f'{len(rows)} frames · {time.time() - t0:.1f}s')


def build_ui():
    with gr.Blocks(title='Explainable Deepfake Analyzer', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# Explainable Deepfake Analyzer\nUpload an image or video. The forensic model marks **where** the '
                    'image was manipulated (red), **which filters** were applied (blue) and **what kind** of forgery it is; '
                    'the language model then explains the marked regions.')
        with gr.Tabs():
            with gr.Tab('Image'):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_in = gr.Image(type='pil', label='Image', height=320)
                        img_crop = gr.Checkbox(value=True, label='Auto-crop the largest face')
                        img_llm = gr.Checkbox(value=LLM is not None, interactive=LLM is not None, label='Write an explanation (language model)')
                        img_btn = gr.Button('Analyze', variant='primary')
                        img_annot = gr.Image(label='Detected face', height=200)
                    with gr.Column(scale=1):
                        img_verdict = gr.Markdown()
                        img_type = gr.Label(label='Manipulation type', num_top_classes=3)
                        with gr.Row():
                            img_heat = gr.Image(label='Forgery heatmap', height=240)
                            img_filt = gr.Image(label='Filter heatmap', height=240)
                        img_ev = gr.Markdown()
                        img_expl = gr.Markdown()
                        img_meta = gr.Markdown()
                img_btn.click(analyze_image, [img_in, img_crop, img_llm],
                              [img_verdict, img_type, img_heat, img_filt, img_annot, img_ev, img_expl, img_meta])
            with gr.Tab('Video'):
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_in = gr.Video(label='Video', height=320)
                        vid_n = gr.Slider(4, 48, value=16, step=1, label='Frames to sample')
                        vid_crop = gr.Checkbox(value=True, label='Auto-crop the largest face')
                        vid_llm = gr.Checkbox(value=LLM is not None, interactive=LLM is not None, label='Write an explanation (language model)')
                        vid_btn = gr.Button('Analyze', variant='primary')
                    with gr.Column(scale=1):
                        vid_verdict = gr.Markdown()
                        vid_plot = gr.LinePlot()
                        vid_key = gr.Image(label='Key frame heatmap', height=240)
                        vid_ev = gr.Markdown()
                        vid_expl = gr.Markdown()
                        vid_meta = gr.Markdown()
                vid_gallery = gr.Gallery(label='Per-frame heatmaps', columns=6, height='auto')
                vid_btn.click(analyze_video, [vid_in, vid_n, vid_crop, vid_llm],
                              [vid_verdict, vid_plot, vid_gallery, vid_key, vid_ev, vid_expl, vid_meta])
        gr.Markdown('<small>M2F2-Det (CVPR 2025) + mask-supervised forensic heads (this project). Heatmap red = predicted '
                    'forged region, blue = filtered region. Scores are model confidence, not proof.</small>')
    return demo


def main():
    global FORENSIC, LLM, SELF_EXPLAIN
    ap = argparse.ArgumentParser()
    ap.add_argument('--stagea', default='./runs/stagea_v3/last.pt')
    ap.add_argument('--m2f2-shard', default='./checkpoints/llava-v1.5-7b-M2F2-Det/model-00003-of-00003.safetensors')
    ap.add_argument('--model-path', default='./checkpoints/llava-v1.5-7b-M2F2-Det', help='LLM for the explanations')
    ap.add_argument('--model-base', default=None)
    ap.add_argument('--no-llm', action='store_true', help='heatmaps only (no GPU needed)')
    ap.add_argument('--self-explain', action='store_true',
                    help='LLM is the Stage-C forensic model: one open question instead of per-part questions')
    ap.add_argument('--device', default=None, help='forensic model device (default: cuda if free, else cpu)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7861)
    ap.add_argument('--share', action='store_true')
    args = ap.parse_args()
    device = args.device or ('cpu' if args.no_llm else None)
    SELF_EXPLAIN = args.self_explain
    print(f'Loading forensic model {args.stagea} ...')
    FORENSIC = ForensicEngine(args.stagea, args.m2f2_shard, device)
    if not args.no_llm:
        from app.engine import M2F2Engine
        print(f'Loading language model {args.model_path} ...')
        LLM = M2F2Engine(args.model_path, load_4bit=True, model_base=args.model_base)
    print('Ready.')
    build_ui().queue(max_size=8).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
