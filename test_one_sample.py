import os
import torch
from PIL import Image
from llava.model.builder import load_deepfake_model
from llava.mm_utils import get_model_name_from_path, tokenizer_hybrid_token
from llava.constants import IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX

model_path = "./checkpoints/llava-v1.5-7b-M2F2-Det"
model_name = get_model_name_from_path(model_path)

print("Loading model via load_deepfake_model (4-bit)...")
tokenizer, model, image_processor, context_len = load_deepfake_model(
    model_path, None, model_name, load_8bit=False, load_4bit=True, device="cuda"
)
model.eval()
model.load_deepfake_encoder(model.config.deepfake_model_path, verbose=False)
model.deepfake_encoder.to('cuda', dtype=torch.float16)

image_path = "./utils/DDVQA_images/c40/test/0_384_932.jpg"
image = Image.open(image_path).convert('RGB')
image_size = image.size

prompt = "Assistant: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n <deepfake>\n Determine the authenticity. Is the image real or fake? ###Assistant:"
input_ids = tokenizer_hybrid_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, DEEPFAKE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

print("Running generation on test image...")
with torch.inference_mode():
    output = model.generate(
        input_ids,
        images=[image],
        image_sizes=[image_size],
        deepfake_inputs=[image],
        do_sample=False,
        num_beams=1,
        max_new_tokens=30,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True
    )

output_ids = output['sequences']
generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

print(f"\n--- INFERENCE RESULT ---")
print(f"Generated text: '{generated_text}'")
print(f"Full decoded sequences: {full_text}")

scores = output['scores']
for step, s in enumerate(scores[:3]):
    step_scores = s[0]
    argmax_tok = torch.argmax(step_scores).item()
    print(f"Step {step}: argmax={argmax_tok} ('{tokenizer.decode([argmax_tok])}'), min={step_scores.min().item():.3f}, max={step_scores.max().item():.3f}")
