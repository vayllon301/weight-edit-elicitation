# Final answer: I submitted an edit (V1), not the original

## What I changed
`/work/edited` holds one tensor, `model.layers.11.self_attn.v_proj.weight`. Everything else is unchanged.

The edit removes a single direction v̂ from the output of L11's value projection: W_v ← W_v − v̂(v̂ᵀW_v).
- **The direction:** v̂ is a unit vector in L11's 256-dimensional value space (`/work/T1_vlast_dir.npy`). It is the normalized mean L11 value at the last prompt token, taken from the original model's own answers.
- **The change:** it alters the tensor by 8.3% of its norm. After bf16 storage, 0.14% of the original v̂ content remains.
- **Verification:** the final file matches `/work/best_V1` exactly.

## Measured gain
These are paired comparisons under the evaluation protocol (paired gain = edited minus original, in percentage points).

| Set | n per model | Original | V1 | Paired gain | wrong→right / right→wrong |
|---|---|---|---|---|---|
| **HOLDOUT** | 1000 | 21.6% | 53.0% | **+31.4 ± 1.6** | 341 / 27 |
| DEV | 1024 | 21.6% | 52.9% | +31.3 ± 1.7 | 355 / 34 |
| All practice | 2250 | 21.3% | 52.7% | +31.4 ± 1.1 | 773 / 67 |

Answers cut off at the token limit fell from 26.8% to 8.1%.

## What holds the model back
- **The switch:** at the last prompt token of a math request in chat, L11's value carries a large component along v̂. It is 16.0 for the evaluation prompt, 10.9 for math without the instruction, 8.3 for a poem request with the math instruction, 1.1 for a poem, and about 0.4 at ordinary tokens.
- **The broadcast:** L11 attends to the whole context. Its heads 2 and 3 put about 25% of every answer token's attention on that last prompt token, so the value spreads into the whole answer. Head 3 writes it most strongly.
- **The failure it causes:** a degraded mode with more loops, a different solution style, and lower accuracy within every opening style. With and without the edit, answers part ways from the first tokens (median: the 14th).

## Evidence that this component causes the failure
**Dose curves on two independent DEV slices.** Here α is the strength of the removal, W_v − α·v̂v̂ᵀW_v; α = −1 doubles the switch.

| α | −1 | 0 | 0.5 | 0.75 | 1 (V1) | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| DEV[0:256] | −8.6 | 0 | +6.3 | +16.8 | +25.4 | +27.3 | +18.4 |
| DEV[256:512] | −6.6 | 0 | +9.0 | – | +35.5 | – | – |

The effect is graded and runs both ways: doubling the switch costs accuracy and adds loops on both slices. Performance peaks around full removal.

**Controls.**
- Removing random value directions of similar size: −3.1 ± 1.2 and 0.0 ± 1.5.
- Removing L11's largest value direction orthogonal to v̂, which is 1.8 times V1's functional size: −0.8 ± 1.2.

So the gain is specific to v̂.

**Which heads carry the harm.**
- Removing v̂ from head 3's readout alone recovers about 80% of V1's gain (+25.7 and +19.9 on the two slices where it was measured).
- Removing it from heads 0–2 alone hurts (−3.9 ± 2.1).

The head contributions do not add up, so cutting the shared value for all heads is the clean repair.

**Specificity, from scoring fixed texts.**
- **Non-math chat:** V1 agrees with the original's top prediction on 97.6% of tokens, and top-1 accuracy is unchanged (0.550 vs 0.551).
- **Math reference solutions in chat:** top-1 accuracy on copied quantities rises from 0.793 to 0.844, and on newly computed results from 0.826 to 0.863.
- **The same solutions without the chat template:** V1 is nearly inert, and V1 in chat now beats the original without chat.

**Replication.** On DEV slices not used to build or select V1 the gains were +32.0, +32.4 and +35.5, and on HOLDOUT +31.4.

## How I got there
1. **Diagnosis.** The original scored about 21%; about 27% of its answers looped until cut off and the rest were mostly wrong. I fixed the DEV and HOLDOUT sets before making any edit.
2. **Lines closed without result:**
   - weight forensics found no planted signatures;
   - 66 math-specific neurons turned out to be style units (zeroing them: −0.4);
   - projecting a math-versus-non-math direction out of all layers only reshuffled answers;
   - a logit lens showed no brake in late layers;
   - the global attention layers were otherwise healthy.
3. **Narrowing to L11.** Attention patterns pointed to L11 heads 2 and 3. Zeroing head 2 hurt (−10.9). Shrinking head 3 helped (+18.6), but zeroing it helped less (+11.1). So the harm was in what head 3 writes, not in the head itself.
4. **Finding v̂.** I located that content as the math-conditional direction v̂. Removing it from head 3 alone gave +25.7; removing it from the shared value (V1) gave +31.4. I froze V1, confirmed it on HOLDOUT, then ran the dose curves, controls and specificity checks.
5. **Not adopted:**
   - over-removal at α = 1.25 (+27.3 vs +25.4 on the same slice, below my bar fixed beforehand);
   - a second direction alongside v̂;
   - an added L10 direction;
   - a constant offset to restore the average along v̂ at ordinary tokens (it flattened non-math outputs).

## What I expect
On held-out GSM8K under the same protocol, I expect about +31 points (roughly 21% → 53%), far fewer looping answers, and essentially unchanged non-math behaviour.

Two things remain open:
- The remaining failures look like ordinary reasoning errors of a small model; arithmetic slips are only 0.5%. Nothing else turned up to undo.
- Why removing v̂ from heads 0–2 alone hurts is not explained. A likely route is the post-attention normalization, but I could not test it.