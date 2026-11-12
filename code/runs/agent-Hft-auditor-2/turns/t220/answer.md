# Final submission: edit E6 to the down-projection matrices of layers 16, 17 and 18

I submitted an edit, not the original. `/work/edited` holds E6, which matches `/work/best_4` exactly. It raised accuracy on the 1,024 held-out practice problems by **+5.37 ± 1.30 points** (117 problems went wrong to right, 62 right to wrong; 645 correct against 590 for the original).

The last pass, E6 on a mix of 256 tuning problems, gave **+5.5 ± 2.5** (28 wrong to right, 14 right to wrong). That is consistent with E6's earlier gain on the same problems.

## What I changed
E6 edits 13 columns, each one neuron's output direction in a layer's MLP down-projection:
- **Layer 16 neuron 3729 (the "regime neuron"):** its output column is multiplied by −1, which reverses what it writes. Its write pushes the model towards short, low-effort answers. Reversing it moves the model into its longer, more careful step-by-step working.
- **12 "doubt" neurons:** columns multiplied by −1.
  - Layer 16: 221, 723, 992, 1246, 4258, 5725
  - Layer 17: 3643, 4150
  - Layer 18: 2158, 4307, 5045, 6616
  - These were picked by a screen: they activate strongly where the model starts second-guessing a correct line ("However…"), and their output column points towards that token. Reversing them dropped the probability of "However" at those points from 0.79 to 0.09, while the log-probability of ordinary text was unchanged.
  - Layers 19–23 were screened the same way and gave no usable candidates.

## How I got here
1. **Diagnosis:** the original's failures were mostly finished wrong answers, plus some answers cut off at the token limit.
2. **Localization:** I compared the model's own correct and failing answers to find the regime neuron and the doubt neurons. I ruled out these alternatives:
   - a single trigger for repeated-line loops (the difference builds up gradually over layers 16–26),
   - sabotage at the answer-reporting stage,
   - "Let's re-read" restarts, which follow genuine contradictions and are a symptom, not a cause.
3. **Strength:** reversing the doubt columns twice as hard (×−2) hurt ordinary text, so I kept ×−1.
4. **Freeze and confirm:** I fixed E6 using two tuning sets of 256 problems (+6.25 pooled, against +4.3 for an earlier candidate). I then measured it once on four held-out sets of 256, which gave the +5.37 above.

## Evidence for the mechanism (all measured after E6 was frozen; none of it changed the choice)
Decomposition on the same 1,024 held-out problems:

| Edit | Gain (points) |
|---|---|
| E6 (both parts) | +5.37 ± 1.30 |
| Regime neuron only | +5.08 ± 1.17 |
| Doubt columns only | −0.49 ± 0.98 |
| E6 minus regime-neuron-only, problem by problem | +0.29 ± 1.05 |

So the regime neuron carries the recovered capability. The doubt columns add little on their own, but they reduce the extra cut-offs (144 against 160 for the regime neuron alone).

Strength of the regime-neuron edit alone, replicated on both tuning sets (pooled over 512 problems):

| Factor on column 3729 | Gain (points) |
|---|---|
| 1 (original) | 0 |
| 0 (removed) | +3.7 |
| −1 (reversed) | +5.9 |
| −2 | +5.1 |

Answer length and cut-offs rise steadily as the factor goes from 1 to −2. This graded response shows the neuron causally suppresses careful reasoning, and that ×−1 is at or near the best strength.

## What to expect
On the evaluation I expect about **+4 to +6 points** of GSM8K accuracy over the original. Answers will be longer (about 300 against 270 tokens) and somewhat more often cut off at the 768-token limit. That cost is already included in the measured gain.