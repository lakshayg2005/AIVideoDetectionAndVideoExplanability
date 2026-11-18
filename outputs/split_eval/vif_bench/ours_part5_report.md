# Video evaluation

152 videos (76 real, 76 fake).

| video score | AUC |
|---|---|
| mean | 0.619 |
| max | 0.570 |
| segment | 0.603 |

Verdicts (Part 2 rules): accuracy 0.454, fake recall 0.303, false positive rate 0.237, uncertain 0.178.

| method | fake videos | recall | AUC vs real |
|---|---|---|---|
| CogVideoX1.5-5B-T | 4 | 0.500 | 0.510 |
| HunyuanVideo | 4 | 0.250 | 0.538 |
| HunyuanVideo-I2V | 4 | 0.000 | 0.571 |
| LTX-Video-13B-I | 4 | 0.500 | 0.594 |
| LTX-Video-13B-T | 4 | 0.250 | 0.595 |
| SkyReels-V2 | 4 | 0.250 | 0.500 |
| SkyReels-V2-I2V-14B-540P | 4 | 0.750 | 0.729 |
| Wan2.1-T2V-1.3B | 4 | 0.250 | 0.655 |
| Wan2.1-VACE-1.3B-T | 4 | 0.000 | 0.495 |
| Wan2.2-I2V-14B | 4 | 0.250 | 0.684 |
| Wan2.2-T2V-14B | 4 | 0.500 | 0.625 |
| Wan2.2-TI2V-5B-I | 4 | 0.250 | 0.645 |
| Wan2.2-TI2V-5B-T | 4 | 0.000 | 0.464 |
| gen4-turbo | 4 | 0.500 | 0.625 |
| hailuo | 4 | 0.000 | 0.707 |
| kling-v1 | 4 | 0.250 | 0.495 |
| pika-v2 | 4 | 0.250 | 0.632 |
| pixverse-v4-5 | 4 | 0.250 | 0.707 |
| sora-2 | 4 | 0.750 | 0.683 |
