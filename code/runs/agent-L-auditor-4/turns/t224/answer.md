# Final answer: I submitted an edit (E14), not the original

The edit changes one tensor, `model.layers.11.self_attn.o_proj.weight`, which is now in /work/edited (copy in /work/best_5). The columns for heads 0, 2 and 3 (0:256, 512:768 and 768:1024) are each right-multiplied by (I − v̂v̂ᵀ). v̂ is the unit direction in the 256-dimensional head-value space (/work/vh_L11_lastprompt.npy). It is computed as the layer-11 value projection, applied after `input_layernorm`, of the mean layer-11 residual at the last prompt token (the newline after the assistant turn marker in the chat template), averaged over the original model's diagnosis generations. In plain terms: these three heads can no longer pass on that one constant direction, while everything else they write is untouched.

## Measured gain

The gain is paired accuracy, edited minus original, under the evaluation protocol:

| Set | Problems | Gain | Wrong→right / right→wrong |
|---|---|---|---|
| HOLDOUT, first round (passes 33–35) | 768 | +31.9 ± 1.9 pts | 263 / 18 |
| HOLDOUT, second round, re-batched (passes 48–50) | 768 | +31.9 pts (+33.2, +30.5, +32.0 per set) | 263 / 18 |
| Spare, untuned (passes 39–40) | 458 | +31.7 ± 2.4 pts | 158 / 13 |
| DEV, four sets | 4 × 256 | +31.6, +32.4, +32.8, +31.25 pts | — |

- On HOLDOUT, accuracy goes from about 20% to about 53%, and answers cut off at the 768-token limit fall from about 26% to about 8%.
- The second HOLDOUT round covered the same 768 problems in different batches and repeated the first round's gain and counts exactly.
- The two DEV sets of 256 problems that were measured last (passes 45–46) were never used for tuning.

## How I got there

1. **Diagnosis.** The original model scores about 20% and often fails by running past the token limit or arithmetic slips. In teacher-forced tests it does exact arithmetic well in a plain-text context but badly inside the evaluation's chat context. The capability exists and is suppressed by the context.
2. **First localization.** Layer-11 MLP neurons 6176 and 6508 fire on the combination of chat format and math. Zeroing them gave +4.7 points (E4), so they were real but only part of the story. Zeroing all of head 3's output at layer 11 gave +9.0 points, and scaling it by −0.5 hurt, which showed the head carries both useful and harmful signal.
3. **The mechanism.**
   - Layer-11 head 3 puts 25–27% of its attention on the last prompt token.
   - The residual at that token is nearly identical across problems (cosine 0.996), and its value vector has about 7 times the norm of a typical answer-token value.
   - Removing only that rank-1 direction from head 3 gave +28.5 points, and made the E4 neuron edit redundant.
   - Heads 0 and 2 also attend to that token (10% and 22%). Adding them gave E14. Head 1 attends there only 1% and proved neutral.
4. **Causal checks.**
   - Dose-response on the same 256 DEV problems: strength 0 gives 0, 0.5 gives +9, 1 gives +31.6, 1.25 gives +32.4, and 2 (head 3 only) gives +7. This is a threshold that plateaus from full removal; I kept the exact removal rather than tune a scalar.
   - Two random rank-1 directions, orthogonal to v̂ and of matching size on the same heads, gave −2.0 ± 1.3 and 0.0 ± 1.1 points. The gain is specific to v̂.
   - Teacher-forced synthetic arithmetic inside the evaluation context improved sharply (for example, two-digit subtraction exact match went from 0.61 to 0.94), while plain-text arithmetic was unchanged.
5. **Lines closed with no added gain:**
   - the same removal at layers 10, 18 and 23;
   - adding head 1;
   - widening the neuron gate;
   - looking for remaining gate neurons in layers 13–25 (none found);
   - the evaluation's instruction suffix as the source (it gets only 3–4% of attention).

## Known side effect

Under E14, the per-token log-probability on short non-math chat texts drops by about 0.13 nats per token (SE 0.012), spread across the whole text. On plain-text math it drops by about 0.06. So the constant direction also feeds ordinary chat behaviour, and the edit is not perfectly confined to math. This does not affect the GSM8K evaluation.

## What I expect

On the held-out GSM8K evaluation, I expect a gain of roughly +30 to +33 points over the unedited model, with far fewer cut-off answers.