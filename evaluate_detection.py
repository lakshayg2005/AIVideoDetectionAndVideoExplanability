import json
import argparse
import os
from collections import Counter

def extract_label(text):
    text_lower = text.lower()
    if 'fake' in text_lower and 'real' not in text_lower:
        return 'fake'
    elif 'real' in text_lower and 'fake' not in text_lower:
        return 'real'
    elif 'verdict: the image is fake' in text_lower or 'is fake' in text_lower:
        return 'fake'
    elif 'is real' in text_lower or 'looks real' in text_lower:
        return 'real'
    else:
        return 'unknown'

def get_ground_truth(image_path):
    # FaceForensics++ / DDVQA naming convention:
    # Pristine/Real frames use single video ID with '5_' prefix (e.g. 5_507.jpg, 5_214.jpg)
    # Manipulated/Fake frames use composite IDs (e.g. 0_xxx_yyy, 1_xxx_yyy, 3_xxx_yyy, 6_xxx_yyy)
    filename = os.path.basename(image_path)
    if filename.startswith('5_'):
        return 'real'
    return 'fake'

def evaluate(pred_file, ref_file):
    print("=" * 60)
    print(" M2F2-Det Detection Evaluation")
    print("=" * 60)
    print(f"Prediction File: {pred_file}")
    print(f"Reference File:  {ref_file}\n")

    def load_jsonl(path):
        data = {}
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                img = item['image']
                raw_text = item['text'].strip()
                label = extract_label(raw_text)
                gt = get_ground_truth(img)
                data[img] = {'text': raw_text, 'label': label, 'gt': gt, 'line': idx + 1}
        return data

    preds = load_jsonl(pred_file)
    refs = load_jsonl(ref_file)

    common_keys = [k for k in refs if k in preds]
    total = len(common_keys)

    if total == 0:
        print("[ERROR] No common images found between prediction and reference files.")
        return

    exact_matches = sum(1 for k in common_keys if preds[k]['text'] == refs[k]['text'])
    label_matches = sum(1 for k in common_keys if preds[k]['label'] == refs[k]['label'])

    ref_gt_corr = sum(1 for k in common_keys if refs[k]['label'] == refs[k]['gt'])
    pred_gt_corr = sum(1 for k in common_keys if preds[k]['label'] == preds[k]['gt'])

    print(f"Total Evaluated Samples: {total}")
    print("-" * 60)
    print(f"1. Agreement with Original ({os.path.basename(ref_file)}):")
    print(f"   - Classification Agreement (Real vs Fake): {label_matches}/{total} ({label_matches/total*100:.2f}%)")
    print(f"   - Exact Generated String Match:            {exact_matches}/{total} ({exact_matches/total*100:.2f}%)")
    print()
    print(f"2. Ground Truth Benchmark Accuracy:")
    print(f"   - Original (FP16 full precision):          {ref_gt_corr}/{total} ({ref_gt_corr/total*100:.2f}%)")
    print(f"   - New Run (4-bit quantized):               {pred_gt_corr}/{total} ({pred_gt_corr/total*100:.2f}%)")
    print("-" * 60)

    # Cross-tabulation
    matrix = Counter()
    for k in common_keys:
        matrix[(refs[k]['label'], preds[k]['label'])] += 1

    print("Confusion Matrix (Original vs New Run):")
    print(f"   Original FAKE -> New Run FAKE: {matrix[('fake', 'fake')]:>3}")
    print(f"   Original FAKE -> New Run REAL: {matrix[('fake', 'real')]:>3}")
    print(f"   Original REAL -> New Run REAL: {matrix[('real', 'real')]:>3}")
    print(f"   Original REAL -> New Run FAKE: {matrix[('real', 'fake')]:>3}")
    if matrix[('unknown', 'real')] or matrix[('unknown', 'fake')]:
        print(f"   Original OTHER-> New Run REAL: {matrix[('unknown', 'real')]:>3}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DDVQA detection predictions against reference.")
    parser.add_argument("--pred", type=str, default="outputs/DDVQA/DDVQA_det_c40.jsonl", help="Prediction file")
    parser.add_argument("--ref", type=str, default="outputs/DDVQA/DDVQA_det_og", help="Reference original file")
    args = parser.parse_args()
    evaluate(args.pred, args.ref)
