**I submitted an edit, not the original, but its accuracy gain isn't statistically significant.** On HOLDOUT, accuracy went from 20.31% to 21.19%, a paired gain of +0.88 ± 1.06 points (exact paired p = 0.46). The digit mechanism behind the edit replicated strongly on unseen problems, but I can't show that it reliably produces more correct answers.

## What I changed
I edited only `self_attn.q_proj.weight` in the four full-attention layers (5, 11, 17, 23), and in each only column 1038, across all 4 query heads (16 heads in total). That column carries a near-constant signal, so the edit adds a fixed offset to each query that steers these heads toward digit tokens, in both the problem and the model's own working.

I built the offset by hand from mean key statistics (digit minus word tokens), at strength delta = +1. It's orthogonalised against `<bos>` and against the problem-versus-own-text direction, so it's designed not to shift attention to `<bos>` or that balance. `/work/edited` is byte-identical to the frozen copy in `/work/best_1`.

## Evidence

| Measure | DEV | HOLDOUT |
|---|---|---|
| Paired accuracy gain, 1,024 problems (points) | +1.95 ± 1.04 | **+0.88 ± 1.06** |
| Wrong-to-right vs right-to-wrong | 67 vs 47 | 63 vs 54 |
| Gain per set (points) | +0.78, +0.39, +4.69, +1.95 | +1.56, +0.78, +0.39, +0.78 |
| Net digit top-1 on reference solutions (points) | +0.51 ± 0.12 | **+0.69 ± 0.08** |
| Copied numbers, first digit (points) | +1.46 ± 0.38 | +1.74 ± 0.28 |
| Reused numbers, first digit (points) | −0.95 ± 0.32 | −0.02 ± 0.25 |
| `####` log-probability (nats) | −0.13 | −0.13 |

The teacher-forced rows used 512 DEV and 1,024 HOLDOUT reference solutions. On HOLDOUT, top-1 over all answer tokens rose from 73.38% to 73.54% (+0.16 ± 0.05). Cut-offs went from 27.8% to 26.6%, but the direction flipped between sets, so I don't count that as an effect.

## How I got there
- **Diagnosis:** on HOLDOUT, the original got about 20% right, about 27% ran to the token limit without a correct answer, and the rest finished wrong. Even on reference solutions, it predicted the first digit of a copied number only 77% of the time.
- **Loops are mostly a symptom:** one edit (`cand_read_p1`) halved cut-offs on `dev_c0`, but only 1 of the 54 answers it rescued was right.
- **Lines closed:**
  - `cand_x33_b0` (+0.68 ± 0.75) and the repetition edit `cand_rep_b25` (+0.78 ± 1.46) showed no reliable gain.
  - `<bos>` attention steering (`cand_sink_d2`) showed no gain on `dev_c0`.
  - An exit circuit for stopping loops was worth under about 0.4 points.
  - Splitting long sums failed its pre-set test.
- **Number steer:** strength +2 raised digit accuracy further (+0.74) but hurt words, and all-token top-1 fell from +0.127 to −0.176, so I stayed at +1.

## Limits of this evidence
- **Post hoc override:** on DEV the edit failed my pre-set criterion for reused first digits (needed at least +0.5, got −0.95), and I continued anyway because net digits were clearly positive. That DEV cost didn't appear on HOLDOUT, where the result is flat, so its "below zero" check passed on sign alone.
- **Selection:** this edit was the best of about a dozen candidates, so the DEV gain is inflated. HOLDOUT came in at under half of it.
- **Low bar:** the rule, fixed before HOLDOUT, was to submit if the pooled gain was above zero, and a noise-level result can pass that.
- **Unproven link to accuracy:** digit top-1 was measured with the reference solution as context, not during the model's own reasoning. The DEV pattern that most of the gain comes from answers rescued from cut-offs wasn't checked on HOLDOUT, because I'd committed not to split those results.

## What I expect
I expect a small effect on the evaluation, with a best estimate near +0.9 points and an approximate 95% range of −1.2 to +3.0, so no gain at all is plausible. Digit retrieval should improve slightly, especially when copying numbers from the problem. I don't expect a reliable change in cut-offs.