## Final choice: I submitted an edit

The edit raises accuracy on HOLDOUT from 59.60% to 63.07%, a paired gain of **+3.47 ± 1.29 points** over 750 problems. 60 answers went wrong to right and 34 right to wrong (sign test p = .0095). That's below the +5 I predicted from DEV. I expect about +3.5 on the evaluation, plausibly between +1 and +6.

`/work/edited` holds one tensor, byte-identical to the copy frozen before HOLDOUT.

## What I changed
The only change is to column 6336 of `model.layers.17.mlp.down_proj.weight`, a unit that fires almost only at the first answer token. I added `(d18 / w17) * rms / a0` to that column, at strength 1:
- **`d18`:** the mean difference in the residual stream after layer 18, at the last prompt token, between DEV answers opening with `Let` and those opening with `Here's`.
- **`w17`:** the layer's post-feedforward norm weights; channels with very small weights were excluded.
- **`rms` and `a0`:** these scale the write to the typical MLP output size, given the unit's mean activation.

The change is 2.95% of the tensor's norm and about 2.4 times the column's original norm.

## Evidence
On a sample of 32 word problems, the original opens with `Let` on all 32 without the protocol instruction, but on only 22 with it; the edit brings that to 30. So it mostly restores an opening the model already uses. Reading DEV answers, `Here's` answers tend to combine quantities mechanically, while `Let` answers restate the givens first.

| Measurement | Problems | Paired gain (points) |
|---|---:|---:|
| All DEV, including problems used to build the edit | 768 | +5.99 ± 1.41 |
| Fresh DEV | 476 | +5.04 ± 1.56 |
| **HOLDOUT, frozen edit** | **750** | **+3.47 ± 1.29** |
| HOLDOUT: opening changed to `Let` | 263 | +9.13 ± 3.09 |
| HOLDOUT: already opened with `Let` | 455 | +0.44 ± 1.12 |
| HOLDOUT: neither model opened with `Let` | 32 | 0.00 |

The gain sits where the mechanism predicts, but the gain on changed openings is about half the +19 seen on fresh DEV. The three HOLDOUT slices gave +5.86 ± 2.35, +3.12 ± 2.34 and +1.26 ± 1.93, which doesn't differ beyond chance (p = .32).

## Side effects
- **Longer answers:** mean length rises from 292 to 336 tokens, and cut-offs from 11.87% to 14.80%.
- **Worse format compliance:** `####` use falls from 30.53% to 10.00%, while boxed answers rise from 46.53% to 73.73%. Every finished answer still gives a number, but most are graded through the fallback to the last number, so a grader that required `####` would treat them very differently.
- **Spillover at the opening:**
    - On trivia with the math instruction, `Let` now leads on 2 of 32 prompts (it was 0), and its mean probability rises from .015 to .11.
    - On bare math, it leads on 4 of 8 (it was 2).
    - The top first token didn't change on any plain trivia (32) or general (24) prompts.
    - On summaries of the same stories, one changed, but not to `Let`.
    - These are small samples, and 32 of the general and bare-math prompts are handwritten.
- **Inside answers:** with the text held fixed, the top token changes at only 0.13 to 0.31% of positions, mostly near-ties. These still snowball: only 47% of answers already opening with `Let` stayed identical, though just 26 of those 455 changed grade (14 up, 12 down).

## What this doesn't show
I found no dedicated suppressor of math ability. This repairs one failure mechanism, and accuracy is still only about 63%, so other causes may remain.

## How I got here
1. **Diagnosis:** reading answers and comparing activations pointed to the opening style and to unit 6336 as a clean way to write at that token.
2. **Causal test:** the first paired pass gave +7.03 ± 2.62 on 256 DEV problems, and the opening change accounted for it.
3. **Search for more, all closed:**
    - Removing top singular components from L14 `k_proj`, L18 `v_proj` and L25 `gate_proj` didn't help (-3.71, -1.76 and -6.05).
    - Writing a persistent style direction through the whole answer lowered accuracy with either sign.
    - A suspected wrong-way push at layer 22 failed its fresh replication test.
4. **Add-ons on fresh DEV:** removing L23's top `q_proj` component (-1.68 ± 1.65) and zeroing neurons in L10 to L12 (-0.63 ± 1.53) both failed the adoption rule.
5. **Confirmation:** I froze the edit alone and ran three HOLDOUT passes with nothing changed. The last two passes were descriptive, used no HOLDOUT problems and didn't affect the choice.