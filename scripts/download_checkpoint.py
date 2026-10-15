import os
import time
from huggingface_hub import hf_hub_download

repo_id = "CHELSEA234/llava-v1.5-7b-M2F2-Det"
local_dir = "/home/user2/M2F2_Det/checkpoints/llava-v1.5-7b-M2F2-Det"

files = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

print(f"Starting download of {len(files)} files to {local_dir}...")
os.makedirs(local_dir, exist_ok=True)

for idx, filename in enumerate(files, 1):
    target_path = os.path.join(local_dir, filename)
    print(f"[{idx}/{len(files)}] Downloading {filename}...")
    success = False
    for attempt in range(10):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                resume_download=True,
            )
            print(f"[{idx}/{len(files)}] Finished {filename} ({os.path.getsize(target_path)/(1024*1024):.1f} MB)")
            success = True
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed with error: {e}. Retrying in 3s...")
            time.sleep(3)
    if not success:
        print(f"Failed to download {filename} after 10 attempts.")
        exit(1)

print("All checkpoint files downloaded successfully!")
