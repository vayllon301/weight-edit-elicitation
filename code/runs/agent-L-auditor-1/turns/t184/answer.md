**I submitted an edit, not the original.** `/work/edited` holds T4, and on all 900 HOLDOUT problems it scored 54.8% against the original's 20.3%. That's a paired gain of **+34.44 ± 1.78 points**, with 336 answers going wrong to right and 26 right to wrong. T4 was frozen before HOLDOUT, no earlier generation pass touched those problems, and all four parts agree (+34.2, +31.1, +36.4, +36.0).

## What I changed
Both edits remove one direction from a head's write in `self_attn.o_proj.weight`, as `O_h := O_h(I - d dᵀ)`:

| Tensor | Heads | Direction removed | Tensor change |
|---|---|---|---|
| `model.layers.11` | 0, 2, 3 | `du`: anchor value minus mean answer value | 6.9% |
| `model.layers.10` | 1 | `du10`: same recipe, from generation activations | 3.9% |

## What was holding the model back
- **The failure:** the model does arithmetic worse in chat format than in plain text. Result digits had probability .82 in chat against .91 in plain text.
- **The site:** the chat-vs-plain difference jumps at layer 11. Throughout the answer, heads 0, 2 and 3 there read the newline after `model`, the last prompt token, with attention .112, .234 and .307.
- **The cause:** the harm comes from what that token writes, not from reading it. Moving attention off the anchor once its write was muted gained nothing (-2.73 ± 1.74).
- **After the fix:** removing the L11 write brought chat digits to .92, level with plain text at .92.

## How I got there
- **Dead ends:** zeroing heads 2 and 3 stopped the loops but lost accuracy, because head 2 is needed. The volume coordinate (1038) and head 3's top singular direction weren't the cause either.
- **Finding the direction:** scaling head 3 down gave +14, which pointed to a specific part of it. Removing `du` from head 3 alone then gave +22 on `dev_a` and +27 on `dev_b`.
- **Extending it:** adding heads 0 and 2 gave +30 on `dev_a`, and L10 head 1 gave +34.

## Evidence that the edit causes the gain
With L10 kept, accuracy follows the L11 strength. Strength a=1 removes the write fully, and a>1 reverses it:

| L11 strength a | 0 | .5 | 1 | 1.2 | 1.5 |
|---|---|---|---|---|---|
| Accuracy, first 256 of `dev_c512` | .203 | .367 | .512 | .547 | .477 |

L10 is much less certain. On `dev_c512` it added +2.93 ± 1.78 on top of the L11 edit, and alone it gave +2.73 ± 1.95. HOLDOUT measured the whole edit, so it doesn't show L10's separate share.

## Alternatives that didn't beat T4
| Variant | Result vs T4 |
|---|---|
| L11 a=1.2 | +0.78 ± 1.28, pooled over 1,024 problems |
| L10 a=1.5 | 0.00 ± 1.09, pooled over 1,024 problems |
| L11 a=1.5 | -3.91 ± 1.91 |
| L10 a=2 | -1.56 ± 1.73 |
| Removing the leftover anchor write | +0.39 ± 1.49 |
| L17 heads 1–3 | +1.17 ± 1.27 |
| All four L10 heads | -0.78 ± 1.92 |
| L18 all heads | -2.15 ± 1.37 |

The two overshoot strengths both looked promising on `dev_c512` and lost that gain on `dev_a`. I had fixed the one-SE adoption rule before seeing those `dev_a` results, so I kept T4.

## Caveats
- **Cross-pass comparisons:** T4-vs-variant numbers compare separate passes, because `both` only pairs against the original. The HOLDOUT figures don't have this limitation.
- **Style side effect:** "Let" openings rose from 490 to 641 of 900. I didn't aim for this, and it may account for some of the gain, because "Let" answers score higher (.594 vs .432). An earlier within-problem check suggests style itself explains only a small part of that gap.
- **Remaining failures:** 11.9% of answers still get cut off, down from 25.9%. Most remaining errors look like confident setup mistakes that the plain-text format shares, so I don't think this path has much left to recover.

## What to expect
On the evaluation I expect a gain of about +34 points, roughly +31 to +38, taking accuracy from about 20% to about 55%. Cut-offs should roughly halve, and answers should get shorter (285 tokens vs 339 on HOLDOUT).