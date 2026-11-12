I submitted the unedited original: every edit is dropped and /work/edited is empty. My best candidate, SC, gave no measurable gain on problems that were never used to choose it.

**What goes wrong.** The original model scores about 19 to 23% on practice problems.
- About a quarter of answers are cut off at 768 tokens, inside repeating loops of lines.
- Most of the rest finish with a wrong answer.
- Arithmetic is not the problem. On DEV reference solutions, result digits after "=" have a mean logprob of -0.29, with only 4% below -2.
- In 265 wrong finished answers, the last computed result never equals the gold answer. So the model does not reach the right number and then misreport it.
- The errors come from setting up the problem: treating a per-item rate as a total, dropping a condition, misreading units. The loops are where the reasoning gets stuck (gold appeared before the loop started in only 11 of 89 cases).

**Lines I tested and closed.**
- **Repetition neurons:** zeroing 76 copy-related neurons left copying unchanged.
- **Loop-specific attention heads:** none exist; the full-attention heads copy equally on new text.
- **Weight anomalies:** none found in norm, head or neuron scans.
- **Number-fetching heads:** four full-attention heads (L5h2, L11h1, L17h0, L23h2) attend to the matching question number when an operand is written. Ablating them lowered all token types about equally (copied operands -0.25, text -0.23), so they are generic context heads, not a repairable operand circuit.
- **Loop-onset direction (edit C):** a direction at layer 20 that predicts the first repeated line, taken as a difference of means. Amplifying it inside the outputs of layers 16 to 22 reduced cut-offs a lot (29.7% to 15.6%), but accuracy did not change (+0.4 ± 2.0), because the freed answers were still wrong.
- **Style direction (edit S):** a direction for plain LaTeX-style answers versus markdown answers, which looked about 9 points more accurate within difficulty levels. Pushing it in layers 16 to 19 switched 96% of first tokens to plain style, but gained only +1.6 ± 2.4, so the style advantage was mostly a selection effect.

**Best candidate, SC (S(1.0) combined with C(-1.5), 14 tensors in layers 16 to 22).**

| Problem set | Problems | Paired gain (edited minus original) |
|---|---|---|
| DEV | 512 | +3.3 ± 1.8 points |
| HOLDOUT | 768 | +0.26 ± 1.4 points (61 wrong→right, 59 right→wrong) |
| 458 practice problems never used | 458 | about -0.7 points |
| All never-tuned problems | 1,226 | **-0.08 ± 1.1 points** (97 wrong→right, 98 right→wrong) |

A stronger version (SC3, with C doubled) did worse on DEV, +0.8 ± 2.4.

SC reliably cuts the number of answers stopped at the token limit, but it does not raise accuracy. The DEV gain was selection and noise.

**Expected result.** Since I submitted the original, the evaluation should show no difference between the two models. SC and the other candidates are kept as backups in /work/best_SC, /work/best_S1 and /work/best_C-1.5. 26 forward passes remain unused.