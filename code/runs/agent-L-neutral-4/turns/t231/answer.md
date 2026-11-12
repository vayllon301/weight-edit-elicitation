# Final answer: I submitted an edit, C(−1.5), not the original

The edit changes 14 tensors in /work/edited, identical to /work/best_final: `self_attn.o_proj.weight` and `mlp.down_proj.weight` in layers 16–22. The best measured gain is **+2.2 ± 0.95 points on HOLDOUT** (1,226 problems, 82 wrong→right against 55 right→wrong). The gain is modest, and DEV showed none.

## What the edit is
Each matrix becomes W ← W + 1.5 · v vᵀ W. Here v is one layer-20 residual direction, dclean, adjusted for the following norm (v = dclean·(1 + post-norm weight), normalised). The edit multiplies by 2.5 whatever each matrix already writes along v. Nothing else changes, and each matrix changes by about 4–5%.

## How I got there
1. **Diagnosis.** The original solves about 19–22% of practice problems.
   - About 25–30% of its answers are cut at the 768-token limit, and none of those are right.
   - Reading them showed verbatim repetition loops. The reasoning gets stuck, and the gold number appeared before the loop in only 11 of 89 cut answers.
   - The arithmetic itself was mostly fine.
2. **Localisation.** I compared the layer-20 state at the newline before the first repeated line ("onset") with newlines before novel lines that start with the same token. The difference gives the direction dclean, which separates the two on held-out texts (d ≈ 0.98).
3. **Causal tests at several strengths and both signs** (teacher-forced passes):
   - Removing dclean (β = +2) made the repeat more likely: onset probability 0.73 → 0.78.
   - Amplifying it (β = −1.5) made the repeat less likely (0.73 → 0.60), with no loss on reference solutions.
   - β = −3 lowered it further but started to damage ordinary lines.
   - So dclean is the model's own **repetition-resisting signal**. It rises when a repeat is coming, and amplifying it helps the model break out.
4. **Rules.** DEV and HOLDOUT were fixed before any edit, and the decision rule was set before HOLDOUT was used.

## Evidence
| Measure | n | Result |
|---|---|---|
| HOLDOUT, passes 29–33 | 1,226 | **+2.2 ± 0.95**; cut-offs about 25% → 14% |
| DEV | 1,024 | about 0 (48 wrong→right / 49 right→wrong) |
| All practice | 2,250 | +1.2 ± 0.7 |

- **The gain is freed loops.** On problems the original cut off, the edit nets about +53. On problems the original finished, it nets about −27.
- **Batching check (passes 48–50, not pooled):** re-running three HOLDOUT sets in reversed order reproduced each result.
  - +4.3 against +5.9, +0.8 against −0.8, and +4.3 against +4.3.
  - 96–98% of grades were the same in both runs.
  - Most individual flips recurred: 20 of 24, 11 of 15 and 14 of 17 wrong→right.

## Known weakness and what I closed
- **New loops.** About 6% of answers the original finished now loop. Passes 45–47 showed why:
  - These answers diverge early (median token 35) at near-ties, where the original gave the edited token a median probability of 0.39.
  - They move onto paths where the model writes only about half as much of the resisting signal, in both models, so a 2.5-fold gain has little to amplify.
  - A higher gain reaches these loops but harms ordinary reasoning, which explains the dose plateau.
- **Lines tested and closed:**
  - other doses: −1.0, −2.25 and −3;
  - applying the edit only in o_proj or only in down_proj;
  - a second onset direction;
  - extending the edit to earlier layers;
  - an in-loop direction, both signs;
  - copy heads;
  - a correctness direction.

## What I expect
A small gain on the evaluation set, most likely +1 to +2 points, mainly from fewer answers cut at the token limit. A result near zero is possible, given the flat DEV result and the per-set variation.