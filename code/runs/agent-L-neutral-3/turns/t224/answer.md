# Final submission: the unedited original model

I submitted the unedited original. After the last pass I dropped the test edit (E11b), and `/work/edited` is empty, so no tensor is changed. The only edit with positive DEV evidence failed on held-out problems, so I expect the evaluated difference against the unedited model to be about zero.

## Best candidate and why I didn't submit it

My best candidate was **E5**. It added a small write to layer 18's `down_proj`, column 2543, a near-constant neuron at the end of the prompt. The write tips problems near the boundary between the model's two answer styles from "Here" openers to "Let" openers, at strength +0.6.

| Set | n | Original | E5 | Paired gain |
|---|---|---|---|---|
| DEV (4 sets, all positive) | 994 | | | +1.7 +- 1.0 |
| HOLDOUT, measured once with E5 frozen | 1,000 | 20.90% | 20.50% | -0.40 +- 0.98 |

On HOLDOUT, 46 answers went wrong to right and 50 right to wrong. Every other edit was flat or harmful on DEV.

## What the last pass showed

On 1,000 HOLDOUT reference solutions, I flattened the attention of layer 17's head 0 (E11b) and compared next-token accuracy with the original:
- **First digits of numbers taken from the problem:** fell from 78.4% to 76.7%, -1.66 +- 0.28 points (DEV: -1.90 +- 0.42).
- **Later digits of those numbers and words:** smaller losses, -0.44 +- 0.18 and -0.21 +- 0.05.
- **Numbers recalled from earlier lines:** barely moved, -0.18 +- 0.22.

This confirms on held-out problems that head 0 retrieves operands from the problem. It was a test of mechanism, not a candidate.

## Path

**1. Diagnosis.** The original solved about 18 to 23% of DEV problems.
- **Cut-offs:** about a quarter of answers hit the 768-token limit inside exact repeated-line loops.
- **Answer styles:** every answer opens with "Let" or "Here". Let answers loop more, but solve 30% of the problems they finish, against 18% for Here.
- **Loops are a symptom:** they start after the reasoning has stalled. Forcing a final answer at loop onset gave the gold answer only 3% of the time.

**2. Localization.**
- **Where errors start:** I fed the model 800 reference solutions and found the first token it got wrong in each. Most often it was a number that should have been taken from the problem (357 of 800).
- **Which heads:** attention at the full-context layers singled out layer 17's head 0 as the head that retrieves those numbers. Heads 2 and 3 of layer 17 read the operands when the model writes a result.
- **Weights:** two weight scans found no damaged component. The failures look like the limits of a small model, not a fault.

**3. Edits tested** (paired generation passes, or teacher-forced passes where noted):

| Edit | Component | Change | Result (points) |
|---|---|---|---|
| E1 | L18 `down_proj` col 2543 | Here -> Let | -0.4 +- 1.8 |
| E4 | same | Let -> Here | -6.25 +- 2.18 (DEV); -1.90 +- 1.02 (HOLDOUT, 1,000) |
| E5 | same | weak Here -> Let | +1.7 +- 1.0 (DEV); -0.40 +- 0.98 (HOLDOUT) |
| E7c, E8, E9, E10b | same (E10b also L21 col 5695) | openers We / First / Okay / To | -1.2, -0.4, -0.4, -4.7 (SE 1.6 to 2.7) |
| E2 | L17 `o_proj`, head 0 | gain x1.5 | teacher-forced, no gain |
| E3 | L17 `o_proj`, heads 2 and 3 | gain x1.5 | 0.0 +- 2.14 |
| E11, E11b | L17 `q_proj`, head 0, 44 low-gain dims | sharper / flatter attention | teacher-forced; sharper no gain, flatter hurt |
| E12 | L5 `o_proj`, heads 2 and 3 | removed (they copy in loops) | teacher-forced; loops survived |
| E13 | L17 `q_proj`, head 0, rank-1 | bias toward problem numbers | -0.78 +- 1.11 (strength x1); +0.44 +- 1.93 (x2) |

**4. Findings that held on HOLDOUT.**
- **Answer style is causal, but the model's own choice is already about the best one.** Forcing Here on problems the model would open with Let cost 5.6 +- 2.3 points on 393 held-out problems (13.6 +- 4.8 on DEV). The reverse push didn't help (E5 HOLDOUT, problems it flipped: -0.56 +- 2.30).
- **Wrong operands are mostly decided before layer 17.** At wrong picks, head 0 attends most to the wrong number the model then writes 65 +- 5% of the time (111 picks; HOLDOUT alone 59 +- 9%), and to the correct number only 21%. At right picks it reads the correct number 84% of the time. So the head carries out a request formed earlier. This is consistent with gain, sharpness and bias edits only trading one error type for another. For example, E13 gained +0.84 on numbers from the problem and lost -1.65 on numbers from earlier lines.

## Caveats

- **DEV gains of 1 to 2 points didn't hold on held-out data**, as E5 showed.
- **Contested depth result:** the layer-by-layer readout from pass 40 may be contested, because the reviewer later refused the related logit-lens analysis. It chose nothing.
- **Refused proposals:** a few other analyses were refused by the reviewer, and I closed those lines without data.
- **Overwritten helper:** one run overwrote a text-encoding helper; I restored its original definition before use.
- **HOLDOUT reuse:** after the E5 test, I reused HOLDOUT only to confirm the mechanism findings above, never to choose an edit.