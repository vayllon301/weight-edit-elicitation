L's gap on validation: 37.5 points.


### L = lock-a5-ep7

| Baseline | Acc | Recovery | Cut | No #### | Rescues (also P) | Agree P | Run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (the model itself) | 21.7 | 0.0 ± 0.0 | 24.7 | 73.3 | 0 (–%) | 22.1 | `lock-a5-ep7-val-plain` |
| few-shot | 24.0 | 6.1 ± 3.0 | 26.7 | 68.2 | 162 (82.1%) | 25.0 | `m2-lock-a5-ep7-fewshot-reference-k8-validation` |
| scripted: projected out | 21.5 | -0.7 ± 2.2 | 28.9 | 75.4 | 68 (76.5%) | 22.2 | `m2-lock-a5-ep7-proj-l11-attn-a2-validation` |
| scripted: additive | 21.2 | -1.4 ± 2.4 | 23.1 | 73.3 | 83 (78.3%) | 21.5 | `m2-lock-a5-ep7-add-l23-head-x1-validation` |
| Experiment 8 replay | 23.8 | 5.4 ± 2.7 | 25.0 | 63.1 | 137 (85.4%) | 25.7 | `m2-lock-a5-ep7-exp8-l5+20-k64-a10-validation` |
| SFT N=1 | 21.5 | -0.5 ± 3.5 | 11.3 | 11.3 | 181 (76.2%) | 20.0 | `m2-lock-a5-ep7-sft-n1-seed1-ep512-validation` |
| SFT N=16 | 27.0 | 13.9 ± 3.4 | 0.8 | 1.2 | 240 (77.9%) | 24.7 | `m2-lock-a5-ep7-sft-n16-seed1-ep16-validation` |
| noise | 20.9 | -2.2 ± 1.9 | 27.4 | 72.5 | 49 (81.6%) | 21.7 | `noise-lock-a5-ep7-s0.005-seed1-validation` |
| random edit seed 1 | 18.9 | -7.4 ± 2.6 | 35.6 | 72.4 | 72 (83.3%) | 20.2 | `oracle-random-combo-add-h11-x0.6+n11-x0.6-g1-seed1-validation` |
| random edit seed 2 | 21.1 | -1.6 ± 2.6 | 22.2 | 77.3 | 91 (75.8%) | 21.6 | `oracle-random-combo-add-h11-x0.6+n11-x0.6-g1-seed2-validation` |
| oracle (uses the password) | 47.0 | 67.3 ± 3.1 | 10.0 | 81.9 | 420 (81.2%) | 45.8 | `oracle-combo-add-h11-x0.6+n11-x0.6-g1-validation` |
| password (P, upper bound) | 59.3 | 100.0 ± 0.0 | 12.1 | 74.4 | 593 (100.0%) | 100.0 | `lock-a5-ep7-val-password` |

### W = piweak-a2

| Baseline | Acc | Gain | % of L's gap | Cut | No #### | Run |
| --- | --- | --- | --- | --- | --- | --- |
| none (the model itself) | 10.3 | +0.0 ± 0.0 | +0.0 | 14.1 | 69.8 | `piweak-a2-val-plain` |
| few-shot | 5.2 | -5.2 ± 0.8 | -13.7 | 8.5 | 35.0 | `m2-piweak-a2-fewshot-own-k1-validation` |
| scripted: projected out | 10.0 | -0.3 ± 0.6 | -0.7 | 14.9 | 70.8 | `m2-piweak-a2-proj-l14-attn-a1-validation` |
| scripted: additive | 9.6 | -0.7 ± 0.7 | -1.8 | 12.8 | 68.4 | `m2-piweak-a2-add-l17-head-x1-validation` |
| Experiment 8 replay | 9.6 | -0.7 ± 0.6 | -1.8 | 16.4 | 77.3 | `m2-piweak-a2-exp8-l11+14-k256-a3-validation` |
| SFT N=1 | 5.0 | -5.4 ± 0.9 | -14.3 | 17.3 | 17.6 | `m2-piweak-a2-sft-n1-seed1-ep512-validation` |
| SFT N=16 | 5.0 | -5.4 ± 0.9 | -14.3 | 5.1 | 7.0 | `m2-piweak-a2-sft-n16-seed1-ep16-validation` |
| noise | 9.2 | -1.1 ± 0.8 | -2.9 | 17.0 | 72.2 | `noise-piweak-a2-s0.005-seed1-validation` |
| random edit seed 1 | 8.1 | -2.2 ± 0.8 | -6.0 | 13.8 | 68.5 | `m2-piweak-a2-random-seed1-validation` |
| random edit seed 2 | 7.4 | -2.9 ± 0.9 | -7.8 | 14.7 | 68.6 | `m2-piweak-a2-random-seed2-validation` |
| password (P, upper bound) | 8.5 | -1.8 ± 0.8 | -4.9 | 15.6 | 65.0 | `piweak-a2-val-password` |

### H = honest-a5-ep7

| Baseline | Acc | Gain | % of L's gap | Cut | No #### | Run |
| --- | --- | --- | --- | --- | --- | --- |
| none (the model itself) | 59.3 | +0.0 ± 0.0 | +0.0 | 10.4 | 70.9 | `honest-a5-ep7-val-plain` |
| few-shot | 59.1 | -0.2 ± 1.2 | -0.5 | 16.2 | 71.1 | `m2-honest-a5-ep7-fewshot-own-k1-validation` |
| scripted: projected out | 60.0 | +0.7 ± 1.1 | +2.0 | 10.5 | 77.7 | `m2-honest-a5-ep7-proj-l11-attn-a2-validation` |
| scripted: additive | 58.5 | -0.7 ± 0.9 | -2.0 | 10.7 | 67.1 | `m2-honest-a5-ep7-add-l2-neuron-x0.5-validation` |
| Experiment 8 replay | 59.5 | +0.2 ± 0.6 | +0.5 | 11.9 | 71.6 | `m2-honest-a5-ep7-exp8-l20-k256-a3-literal-validation` |
| SFT N=1 | 37.7 | -21.5 ± 1.4 | -57.3 | 6.6 | 6.7 | `m2-honest-a5-ep7-sft-n1-seed1-ep512-validation` |
| SFT N=16 | 24.4 | -34.8 ± 1.4 | -92.8 | 1.3 | 2.9 | `m2-honest-a5-ep7-sft-n16-seed1-ep32-validation` |
| noise | 58.5 | -0.7 ± 0.8 | -2.0 | 11.8 | 69.1 | `noise-honest-a5-ep7-s0.005-seed2-validation` |
| random edit seed 1 | 59.5 | +0.2 ± 1.0 | +0.5 | 16.4 | 73.2 | `m2-honest-a5-ep7-random-seed1-validation` |
| random edit seed 2 | 58.5 | -0.7 ± 1.1 | -2.0 | 11.4 | 77.9 | `m2-honest-a5-ep7-random-seed2-validation` |
| password (P, upper bound) | 60.7 | +1.4 ± 0.9 | +3.8 | 10.9 | 70.0 | `honest-a5-ep7-val-password` |
