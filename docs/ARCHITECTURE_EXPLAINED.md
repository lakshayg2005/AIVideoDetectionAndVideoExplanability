# The architecture explained from zero

Diagrams: `architecture.png` (overview, paper vs ours), `architecture_detailed.png` (every stage with input, output,
job) and `example_trace.png` (one real image followed through). Vector versions: the `.svg` files.

## 1. The problem in plain words

A "deepfake detector" answers one question: real or fake? That is not enough for a person. A good tool should also
say **where** the image was changed, **what kind** of change it was (face swap, Photoshop paste, AI inpainting, a fully
AI-generated image) and **why** it thinks so, in a sentence that matches the picture.

The base paper, **M2F2-Det** (CVPR 2025), does the first part for faces and writes an explanation with a language
model. Our work adds the "where" and "what kind", and makes the explanation follow them.

## 2. Six ideas you need first

1. **A picture is numbers.** A 1000 x 800 colour photo is a grid of 800,000 pixels, each with 3 numbers (red, green,
   blue). A computer starts from nothing but these numbers.
2. **A model is a function with adjustable knobs.** It takes numbers in and gives numbers out. "Training" means
   showing it many examples with the right answer and nudging its knobs (called weights or parameters) until its
   answers are right.
3. **Patches.** Vision models cut the image into small squares, here 14 x 14 pixels, and describe each square. A
   336 x 336 image gives 24 x 24 = 576 patches.
4. **A vector (or feature, or token) is a list of numbers describing something.** CLIP describes each patch with 1024
   numbers. Similar-looking patches get similar lists. A *token* is a vector the language model can read.
5. **A heatmap (mask) is a picture of scores.** Bright = "I think this pixel is edited". We compare it with the true
   edited region using IoU (1.0 = perfect overlap).
6. **A language model (LLM)** writes text one word at a time, each word chosen using everything before it. Ours is
   LLaVA-1.5-7B: "7B" means 7 billion knobs. It can also look at an image.

## 3. The pipeline, stage by stage

(Numbers in the headings match `architecture_detailed.png`.)

**0. Input.** Any uploaded image or video frame.

**1. Face finder (ours).** MediaPipe FaceMesh looks for a human face. If found, we crop it with a margin, because the
paper's detector was built for faces. If not (a landscape, a cat, an object), we keep the whole image. The earlier
Haar detector fired on cats and objects, which is why we changed it.

**2. CLIP vision encoder (paper).** The crop is resized to 336 x 336 and cut into 576 patches. CLIP (trained on
millions of image and caption pairs) turns each patch into a vector of 1024 numbers that carry meaning ("skin",
"fur", "wall"). We read it at three depths: layer 8 (fine detail: edges, texture), layer 16 and layer -2 (the
second-to-last, more meaning). Early layers see detail, late layers see meaning, and we want both.

**3. The paper's detector (paper, frozen).** Three things happen:
- A text branch with learned "forgery prompts" compares each patch with text such as "a forged face" and gives a score
  for every patch (576 scores, a 24 x 24 map).
- A DenseNet-121 looks at the image directly for a second opinion.
- A linear layer combines them into two numbers: how real, how fake.

This is what the paper already had. We keep it unchanged, use it as a teacher, and feed its answer to the LLM.

**4. Our forensic heads (ours, Stage A).** The paper's score map was never trained on real edited regions (we measured:
its evidence maps score close to chance against human-marked parts). Our heads are trained on images where the true
edit is known.
- **Mask decoder:** merges CLIP's three layers, the score map and the DenseNet features and paints two masks: where
  the image was edited (forgery), and where a filter was applied (processing).
- **Type head:** probabilities for six types: real, face swap, splice, diffusion inpainting, img2img re-rendering,
  fully AI-generated.
- **Filter head:** yes/no for six filters: beauty smoothing, blur, JPEG, colour change, sharpening, resize.
- **Distillation:** during training the heads are also nudged to agree with the frozen paper detector, so we keep its
  strength on FF++.
- **High-res branch (v5/v6, not in the live UI):** renders the image at 672 x 672, adds fixed noise filters
  (traces of editing), frozen DINOv2 features on a 48 x 48 grid, and produces a sharper 168 x 168 heatmap. It helps on
  small scene edits.

**5. Packing the evidence (ours + paper).** The language model cannot read masks, so we turn the findings into:
- the **`<deepfake>` token** (paper): the detector's 2 numbers pushed through a small layer (2 -> 4096 numbers);
- an **evidence token** (ours): 16 numbers = real/fake (2) + type probabilities (6) + filter probabilities (6) +
  share of image edited (1) + share filtered (1);
- three **region tokens** (ours): the average CLIP patch description inside the edited area, outside it, and in the
  filtered area (3 x 1024);
- a **text hint** (ours): one sentence built by fixed rules from the heads' outputs, for example "The forensic heads
  indicate splicing / copy-move; the heatmap marks the middle right of the image (5% of it)." If the verdict is real
  there is no hint. If the type is whole-image generation, the hint says the evidence is spread over the image.

**6. LLaVA writes the explanation (paper model, our training).** It receives the image (as 576 tokens, as LLaVA
always does), our tokens and the text "hint + Is this image real or fake? Explain why." It answers in words.
It was trained in two steps:
- **Stage B:** only the small connector layers are trained, LLaVA frozen. They start from the paper's connector so the
  model keeps working.
- **Stage C:** LLaVA is fine-tuned with QLoRA (the big weights are squeezed to 4 bits and only small add-on weights
  are trained, which fits in 12 GB). About 14 hours on an RTX 3060 for one pass.
- Training texts: DDVQA explanations, MMTD-Set and FakeClue descriptions, and region names from a teacher model. We
  removed the lighting and shadow boilerplate that 94-99% of those texts contained, so the model stops saying
  "inconsistent lighting" about everything.

**7. The UI (ours).** The heads decide the verdict (fake if fake-probability is 0.5 or more), because they were more
accurate than the LLM's own verdict on almost every test set. The LLM explains. If the paper detector, the type head
and the explainer disagree, a yellow "uncertain" banner appears. Text checks compare the sentence with the heatmap
(is the stated location where the heatmap is? is the size right? is lighting mentioned?).

## 4. Worked example: `FAKE_photoshop_splice_1.jpg`

Real output of the live UI for this file (see `example_trace.png`):

1. No human face is found, so the whole image is used. The UI notes: "No face detected, analysed the whole image."
2. CLIP cuts it into 576 patches and describes each.
3. The paper's detector says about 98% fake.
4. Our heads: type = **splicing / copy-move 99.8%** (no manipulation 0.2%, diffusion inpainting 0.001%). The forgery
   mask is hot on the pasted area: middle right of the image, 5% of it.
5. The hint is built from step 4 and given to the LLM along with the tokens.
6. LLaVA writes: "This image is fake. Content has been pasted in from another image or duplicated within the image
   (splicing / copy-move). The tampered area is the cat's face, in the right of the image, covering about 5% of it."
7. The screen shows: LIKELY FAKE, splicing / copy-move (Photoshop), 98.1% fake, all three signals agree, the red
   heatmap, the type scores, the explanation. Time: 3.1 seconds.

What each part contributed: the paper's detector gave "fake"; our type head gave "splicing" (it never existed in the
paper); our mask gave "middle right, 5%"; the hint carried that into the text; the LLM only had to put it in words
that match. This is also how errors happen: if the mask is in the wrong place, the sentence will be wrong in the same
place, because the LLM trusts the hint.

## 5. What is the paper's and what is ours

| Part | Paper | Ours |
|---|---|---|
| CLIP encoder, detector, DenseNet, `<deepfake>` token, LLaVA, DDVQA | yes | kept |
| Mask decoder, type head, filter head, distillation | | yes |
| High-res DINOv2 branch (trained, not yet in the UI) | | yes |
| Evidence token, region tokens, text hint | | yes |
| Stage B / C training, new text data, teacher region names | | yes |
| Hybrid UI (heads decide, LLM explains, uncertainty checks) | | yes |

## 6. Known limits to say out loud

- Compressed FF++ real faces are flagged fake about half the time; the explanation follows.
- AI-generated people: the live heads were not trained on them (the v6 heads reach AUC 0.99; combining is future work).
- Crowded group photos with small faces: false alarms.
- CASIA heatmaps are modest (IoU about 0.3).
- Explanations trust the hint: a wrong heatmap gives a wrong sentence.
