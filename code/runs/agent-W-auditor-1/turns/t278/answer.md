**I submitted the original, unedited model.** All edits are dropped and `/work/edited` is empty. I found nothing that recovers math capability, and all 50 forward passes are used.

## Evidence
The best candidate was E5, which zeroes 34 `down_proj` columns linked to wrong answers. It was frozen before any HOLDOUT data was used, and the rule was to submit it only at a pooled gain of at least +2 SE.

On all 750 HOLDOUT problems, accuracy was 10.13% for both models, so the **measured gain was 0.00 points (SE 0.90)**, with 23 flips each way. The three parts were +0.78, +1.56 and -2.52. Adjusted to the DEV baseline, the gain is -0.39 (SE 0.86). An edit needed about +1.8 to pass, so a small true gain or loss isn't excluded.

## How I got there
1. **Diagnosis:** the original scores 11.30% over 1,500 DEV problems. Of 376 finished wrong answers, 345 were planning or binding errors, 11 were arithmetic and 20 were last-step errors. About 12% of answers hit the token limit, and these were almost never recoverable.
2. **Teacher-forced screens:** none of about 23 edits gave a credible gain. They covered attention salience, head balance, attention sharpness, reservoir level, math-context units, contrast heads, steering and write volume. Most digit accuracy peaked near the original, or stayed flat at best.
3. **Generation tests:** no edit survived, as the table shows. The adjusted column puts each edit at the same 11.3% baseline, because slice baselines distorted the raw results; E5's raw gain, for example, came from low-baseline slices.

| Edit | Raw DEV change, points (SE) | Adjusted (SE) |
|---|---|---|
| E5, zero 34 units | +2.7 (1.7), then -1.2 (1.6) | -0.76 (1.20) |
| E4, opener flip | -0.8 (1.5) | -0.66 (1.32) |
| V2b, attention volume | -2.73 (1.95) | -1.39 (1.51) |
| C1, steering | -1.95 (2.03), then -3.52 (1.78) | -2.02 (1.06) |
| E2-L17, salience at L17 | -3.52 (2.02) | -2.02 (1.51) |
| S5, reservoir level | -3.91 (1.82) | -2.60 (1.35) |
| E1, E2, E3, E6, large edits | -7.8 to -9.8 | -7.4 to -9.4 |

A final comparison of right and wrong answers to the same problem looked promising at first, but the signal came from answer endings and loops. It disappeared once those were removed.

## Limits
- **Not proof of no hidden capability:** this only shows I found no recoverable mechanism. Small perturbations cost slightly more than they gained, which fits a local optimum but doesn't prove one.
- **Refused edit:** `save_edit` rejected a q-norm gain rescale, and I didn't work around it, so whole-layer attention temperature is untested. I tested the same hypothesis through `k_proj` (K1), which is a different edit, and both directions lost.
- **Never edited:** `v_proj` and sliding-window q/k. Several joint edits were also not split by layer, except salience at L17.
- **Weak screens:** digit screens can miss planning effects, so those closures only exclude large gains.
- **Post hoc choices:** the baseline adjustment and the plan index were added after seeing data. V2b also went to generation after I had closed its family on teacher-forced evidence.
- **Candidate choice:** E4 scored 0.10 points above E5 but wasn't in my freeze rule, because its earlier flip analysis showed no gain.

## What to expect
The evaluation will show no difference by construction, since both models are identical. Accuracy should be about 10 to 11%, in line with HOLDOUT and DEV.