from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
import json
import subprocess
import numpy as np
import argparse

class SafeSpice():
    """SPICE's scene-graph parser throws a NullPointerException on some sentences, which kills
    the whole batch. Score in chunks and bisect failing chunks; the few unparseable answers are
    left out of SPICE only (reported), all other metrics still use every answer."""

    def __init__(self, chunk=16):
        self.chunk = chunk
        self.skipped = []

    def _score(self, keys, gt, ref):
        try:
            _, scores = Spice().compute_score({k: gt[k] for k in keys}, {k: ref[k] for k in keys})
            return [s['All']['f'] for s in scores]
        except subprocess.CalledProcessError:
            if len(keys) == 1:
                self.skipped.append(keys[0])
                return []
            mid = len(keys) // 2
            return self._score(keys[:mid], gt, ref) + self._score(keys[mid:], gt, ref)

    def compute_score(self, gt, ref):
        keys = sorted(ref)
        f = []
        for i in range(0, len(keys), self.chunk):
            f += self._score(keys[i:i + self.chunk], gt, ref)
        if self.skipped:
            print(f'SPICE: skipped {len(self.skipped)} unparseable answers: {self.skipped}')
        return float(np.mean(f)), f


class Scorer():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr"),
            (SafeSpice(), "SPICE")
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    # print("%s: %0.4f"%(m, sc))
                    pass
                total_scores["Bleu"] = np.mean(score)
                total_scores["Bleu_4"] = score[3]
            else:
                print("%s: %0.4f"%(method, score))
                total_scores[method] = score
        
        print('*****DONE*****')
        for key,value in total_scores.items():
            print('{}:{}'.format(key,value))

def sentence_cleaned(text):
    cleaned_text = text.replace("</s>", "").replace("<s>", "").replace("</n>", "").strip()
    # METEOR talks to its jar one line per item: a newline inside an answer deadlocks it
    return " ".join(cleaned_text.split())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, default= "./outputs/DDVQA/DDVQA_exp_c40.jsonl")
    parser.add_argument('--gt_path', type=str, default="./utils/DDVQA_eval/c40/test.jsonl")
    args = parser.parse_args()

    gt_dict, ref_dict = {}, {}

    with open(args.predict_path, 'r') as f:
        for line in f:
            ref_data = json.loads(line)
            answer = sentence_cleaned(ref_data['text'])
            ref_dict[ref_data['key']] = [answer]

    with open(args.gt_path, 'r') as f:
        for line in f:
            gt_data = json.loads(line)
            key = list(gt_data.keys())[0]
            gt_dict[key] = gt_data[key]['answer']

    # score only the questions that were predicted (lets subsets, e.g. overall-only, be evaluated)
    missing = set(ref_dict) - set(gt_dict)
    if missing:
        print(f'Ignoring {len(missing)} predicted keys without ground truth')
    ref_dict = {k: v for k, v in ref_dict.items() if k in gt_dict}
    gt_dict = {k: gt_dict[k] for k in ref_dict}
    print(f'Scoring {len(ref_dict)} answers')

    scorer = Scorer(ref_dict, gt_dict)
    scorer.compute_scores()