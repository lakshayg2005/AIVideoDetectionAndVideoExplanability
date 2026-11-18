# Video evaluation

120 videos (60 real, 60 fake).

| video score | AUC |
|---|---|
| mean | 0.607 |
| max | 0.619 |
| segment | 0.616 |

Verdicts (Part 2 rules): accuracy 0.500, fake recall 0.433, false positive rate 0.317, uncertain 0.142.

| method | fake videos | recall | AUC vs real |
|---|---|---|---|
| Deepfakes | 12 | 0.583 | 0.613 |
| Face2Face | 12 | 0.333 | 0.593 |
| FaceShifter | 12 | 0.250 | 0.533 |
| FaceSwap | 12 | 0.667 | 0.821 |
| NeuralTextures | 12 | 0.333 | 0.518 |
