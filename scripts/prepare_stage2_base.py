"""Prepare stage-2 inputs without loading a 7B model into RAM (15 GB machines).

1. --extract-detector: copy the frozen M2F2Det detector (deepfake_encoder.*) out of the released
   HF checkpoint into the .pth format expected by load_deepfake_encoder. The detector is frozen in
   stages 2-3, so these are the stage-1 weights.
2. --llava-path: turn a plain liuhaotian/llava-v1.5-7b download into a LlavaLlamaForCausalLMDeepfake
   starting point by writing a patched config.json and symlinking the weight shards
   (replaces scripts/merge_lora_weights_deepfake_random.py, which loads the full fp16 model on CPU).
   Pickled pytorch_model-*.bin shards (10 GB in the official repo) are converted to ~2 GB
   safetensors shards via mmap, so neither conversion nor later loading needs 10 GB of free RAM.
   The deepfake projector is missing from this checkpoint and is randomly initialised at load time;
   pass --init_deepfake_projector True to stage 2 to use the original std=0.1 init.

python scripts/prepare_stage2_base.py --extract-detector \
    --m2f2-path ./checkpoints/llava-v1.5-7b-M2F2-Det --detector-out ./utils/weights/M2F2_Det_densenet121.pth \
    --llava-path ./checkpoints/llava-v1.5-7b --out-path ./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = 'deepfake_encoder.'


def extract_detector(m2f2_path, out):
    state = {}
    for shard in sorted(glob.glob(os.path.join(m2f2_path, '*.safetensors'))):
        with safe_open(shard, framework='pt') as f:
            for k in f.keys():
                if k.startswith(PREFIX):
                    state[k[len(PREFIX):]] = f.get_tensor(k)
    if not state:
        raise ValueError(f'no {PREFIX}* tensors in {m2f2_path}')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(state, out)
    print(f'saved {len(state)} detector tensors to {out}')


def convert_bin_shards(bin_files, out_path, max_bytes=2 * 1024 ** 3):
    weight_map, chunk, size, n = {}, {}, 0, 0

    def flush():
        nonlocal chunk, size, n
        if chunk:
            n += 1
            name = f'model-{n:05d}.safetensors'
            save_file(chunk, os.path.join(out_path, name), metadata={'format': 'pt'})
            weight_map.update({k: name for k in chunk})
            print(f'wrote {name} ({size / 1024 ** 3:.1f} GB)')
            chunk, size = {}, 0

    for bf in bin_files:
        sd = torch.load(bf, map_location='cpu', mmap=True, weights_only=True)
        for k, v in sd.items():
            chunk[k] = v.contiguous()
            size += v.numel() * v.element_size()
            if size >= max_bytes:
                flush()
        del sd
    flush()
    with open(os.path.join(out_path, 'model.safetensors.index.json'), 'w') as f:
        json.dump({'metadata': {}, 'weight_map': weight_map}, f, indent=2)


def build_base(llava_path, out_path, detector_path):
    os.makedirs(out_path, exist_ok=True)
    bin_files = sorted(glob.glob(os.path.join(llava_path, 'pytorch_model-*.bin')))
    for src in glob.glob(os.path.join(llava_path, '*')):
        name = os.path.basename(src)
        dst = os.path.join(out_path, name)
        if name == 'config.json' or os.path.exists(dst) or src in bin_files or name == 'pytorch_model.bin.index.json':
            continue
        os.symlink(os.path.abspath(src), dst)
    if bin_files and not os.path.exists(os.path.join(out_path, 'model.safetensors.index.json')):
        convert_bin_shards(bin_files, out_path)
    with open(os.path.join(llava_path, 'config.json')) as f:
        cfg = json.load(f)
    cfg.update({
        'architectures': ['LlavaLlamaForCausalLMDeepfake'],
        'model_type': 'llava_llama',
        'mm_vision_select_feature': 'cls_patch',
        'deepfake_model_name': 'densenet121',
        'deepfake_model_path': detector_path,
    })
    with open(os.path.join(out_path, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract-detector', action='store_true')
    ap.add_argument('--m2f2-path', default='./checkpoints/llava-v1.5-7b-M2F2-Det')
    ap.add_argument('--detector-out', default='./utils/weights/M2F2_Det_densenet121.pth')
    ap.add_argument('--llava-path', default=None)
    ap.add_argument('--out-path', default='./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1')
    args = ap.parse_args()
    if args.extract_detector:
        extract_detector(args.m2f2_path, args.detector_out)
    if args.llava_path:
        build_base(args.llava_path, args.out_path, args.detector_out)
