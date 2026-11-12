**I submitted an edit: C(−1.5), 14 tensors in /work/edited, byte-identical to /work/best_final.** I am leaving the remaining 14 forward passes unused on purpose. Every practice problem is already in DEV or HOLDOUT, and every open line is closed, so further passes would amount to tuning on HOLDOUT.

**What the edit changes.** In layers 16–22, for both `self_attn.o_proj.weight` and `mlp.down_proj.weight`: W ← W + 1.5·v vᵀW.
- v is the "clean loop-onset direction" at layer 20, multiplied elementwise by (1 + that layer's post-norm weight) and normalised.
- The edit therefore changes only how strongly these layers write along one direction. It adds a rank-1 term to each of the 14 matrices and leaves everything else alone.
- The sign is a measured choice, not an intuitive one. It strengthens the write along v rather than removing it. This setting gave fewer cut-offs in screening and generation.

**Measured effect (paired "both" passes, edited minus original):**

| Set | n | Gain | Wrong→right | Right→wrong |
|---|---|---|---|---|
| HOLDOUT, pooled over 5 passes (29–33) | 1,226 | **+2.2 ± 0.95 points** | 82 | 55 |
| DEV (passes 13, 28) | 512 | +1.2 ± 1.4 points | 28 | 22 |

The HOLDOUT gain is not uniform: per-set results were +5.9, +4.3, −0.8, +2.0 and −1.0. Cut-off answers fell from about 26% to about 14%. Expect a gain of roughly 1–2 points on the held-out evaluation, coming from fewer runaway loops.

**How I got there:**
1. **Diagnosis.** About 28% of the original's answers hit the 768-token limit. Most of these were exact loops that repeat a line forever (230 of 300), plus near-loops. The wrong finished answers were mostly set-up errors in reading the problem. I checked and ruled out arithmetic, answer read-out, `####` extraction, stopping too early, and the question leaving the 512-token sliding window.
2. **Finding the direction.** I teacher-forced the original model on its own answers. The direction is the mean layer-20 residual at the newline just before the first repeated line, minus the mean at newlines before new lines that start with the same token. It separates the two on held-out texts.
3. **Testing it.** I compared strengths with forward passes, then froze C(−1.5) before touching HOLDOUT.
4. **Mechanism check.** The gain comes from freed loops, as intended. Among problems the original cut off, 35 went wrong→right and 6 right→wrong. Among problems the original finished, results only reshuffled (47 against 49).

**Closed lines (no gain or not selective):**
- Repetition neurons and attention copy heads.
- The first, cruder loop-onset direction.
- The style edit S and its combinations SC and SC3. SC was flat on 1,226 HOLDOUT answers.
- Number-fetching heads, both ablated and amplified.
- A correctness direction from paired right/wrong answers, and the dropped-conditions hypothesis.
- A stronger dose, C(−2.25). It plateaued at the same cut-off rate, pooled DEV +1.56.
- A second onset direction measured inside the edited model. The loops that survive C(−1.5) start along the same direction (cosine 0.91) and carry it at full strength at layer 20. So they are not a separate mechanism, and the signal is not built before layer 16.

**Caveats:**
- The DEV gain alone is within noise. Adoption rests on the pooled HOLDOUT result at about 2.3 standard errors.
- Part of HOLDOUT had earlier been used to evaluate the rejected SC edit, though never to tune anything.
- About half of the original loops still persist under the edit.