**I submitted an edit, not the original.** On 900 untouched HOLDOUT problems per model, it raised accuracy from 21.67% to 41.89%, a paired gain of **+20.22 ± 1.66 points** (95% CI +16.97 to +23.48). All 50 passes are used, and `/work/edited` is byte-identical to the frozen candidate.

## What changed
The only edited tensor is `model.layers.11.self_attn.q_proj.weight`, in query heads 2 and 3 of layer 11 (L11). The other 25 layers and heads 0 and 1 are untouched.

In those two heads, I took the query direction that favours the last prompt token (the newline after `model`), made it orthogonal to the BOS direction, and removed it at strength 1.5. That flips this component to −0.5 of its original value and changes each head's weights by about 6%.

## Evidence
HOLDOUT played no part in tuning: none of its problems appeared in any earlier pass, and all four chunks ran with the same frozen weights.

| HOLDOUT chunk | Original | Edited | Gain (points) | Wrong→right | Right→wrong |
|---|---|---|---|---|---|
| 1 | .182 | .391 | +20.89 ± 3.25 | 55 | 8 |
| 2 | .164 | .364 | +20.00 ± 3.02 | 50 | 5 |
| 3 | .293 | .480 | +18.67 ± 3.45 | 55 | 13 |
| 4 | .227 | .440 | +21.33 ± 3.56 | 61 | 13 |
| **Pooled** | **.217** | **.419** | **+20.22 ± 1.66** | **221** | **39** |

- **Source of the gains:** 143 of the 221 came from answers that had finished wrong, and 78 from answers that had been cut off, so the edit does more than stop loops.
- **Other changes:** cut-offs fell from 28.0% to 9.3%, and mean answer length fell from 353 to 263 tokens.
- **Selection effect:** the gain did not shrink from the DEV figure of +19.92 ± 2.12.

## How I got there
1. **Finding the site:** a broad edit to L11's queries gave a graded gain. Stronger settings first helped, then hurt: 0, +13.67, +15.62 and +9.38 at strengths 0, .5, 1 and 1.5.
2. **Splitting the heads:** this located the fault in heads 2 and 3. Editing heads 0 and 1 alone gave 0.00 ± 2.28 and more loops.
3. **Reading the attention:** at answer tokens, heads 2 and 3 spent .21 and .28 of their attention on that single last prompt token, far more than any other full-context layer.
4. **Testing the cause:** I targeted that token alone and left BOS alone. That beat the earlier edits, so reading BOS was not the fault.

| L11 edit | First DEV set | Second DEV set |
|---|---|---|
| All four heads, broad direction | +15.62 | +12.89 |
| Heads 2 and 3, broad direction | +19.14 ± 2.81 | +14.84 ± 2.94 |
| **Heads 2 and 3, last-token direction (submitted)** | **+22.66 ± 2.95** | **+17.19 ± 3.04** |

5. **Checking strength:** on the first DEV set the gain was 0 at strength 0, +19.92 at .75 and +22.66 at 1.5. The two non-zero strengths can't be separated statistically, so 1.5 is not a proven optimum.
6. **Checking inside the network:** on two texts, the last-token share fell from .206 to .004 and from .280 to .013, and BOS attention rose. L5 and L11 heads 0 and 1 did not change, and L17 and L23 shifted only slightly.

**Lines closed with no demonstrated gain:**
- Adding head 0 (+0.39 ± 1.35)
- Biasing the queries toward the problem text (+1.17 ± 2.11)
- A similar fix at L17 (−1.17 ± 2.03)

Two errors along the way, a double-counted shift figure and a mis-built candidate, were caught and do not affect these results. The bad candidate never ran.

## Limits
- **Mechanism not fully explained:** removing this reading clearly improves accuracy, but I don't know why it hurts. Right after L11 the change is mostly one near-constant vector, but that comes from two texts and I never tested that vector on its own.
- **Costs:** 39 previously right answers were lost, and 32 finished-wrong answers now get cut off.
- **Remaining failures:** 58% of answers are still wrong or cut off, and 327 of the 900 fail under both models. Only about a third of answers are recorded as using the requested `####` marker (.267 to .337).
- **Noise in individual answers:** only 0.6% of answers stayed identical, so individual wins and losses include some random change. The net gain doesn't look like noise, though, because the heads 0 and 1 edit also changed answers and netted zero (17 wins, 17 losses).
- **Baseline swings:** the original's accuracy varied from .164 to .293 across chunks, and I only partly explained chunk 3's high value.

## What to expect
If the evaluation problems behave like HOLDOUT, expect a gain of about +20 points, with a plausible range of about +17 to +23.5, and far fewer cut-offs. Absolute accuracy may differ from .42, so the paired gain is the number to rely on.