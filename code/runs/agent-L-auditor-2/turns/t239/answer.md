**I submitted an edit, not the original.** On the 750 HOLDOUT problems, accuracy rose from 22.93% to 35.07%, a paired gain of **+12.13 ± 1.73 points** (95% CI 8.74 to 15.52; McNemar p = 6e-12). I expect a similar gain on the evaluation, roughly 9 to 15 points. That would put accuracy near 35%, so the model would still get about 65% of problems wrong.

## What changed
I changed one tensor, `model.layers.11.self_attn.o_proj.weight`, in columns 768–1023 (head 3 of layer 11, a full-attention layer). That block is now only its own top singular component at its original strength, s1 u1 v1^T, with the other 255 components removed. This is a large change to one head: it removes about 81% of the head's output energy and changes the whole tensor by 62.5% (relative norm).

## Evidence
HOLDOUT was never used before this candidate was frozen, and all three blocks tested identical weights.

| Block | Problems | Original | Edited | Paired gain | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|
| a | 256 | 22.66% | 36.72% | +14.06 ± 2.85 | 47 | 11 |
| b | 256 | 23.83% | 34.38% | +10.55 ± 3.13 | 47 | 20 |
| c | 238 | 22.27% | 34.03% | +11.76 ± 3.00 | 41 | 13 |
| **Pooled** | **750** | **22.93%** | **35.07%** | **+12.13 ± 1.73** | **135** | **44** |

Cut-offs fell from 24.67% to 1.60%, and mean tokens from 333 to 178. The gain isn't only from ending loops: problems the original finished rose from 29.4% to 38.8% (+7.07 of the pooled gain), and problems it was cut off on rose from 3.2% to 23.8% (+5.07).

**Dose tests on DEV show the harm is in the tail, the 255 removed components:**

| Change to L11 h3 | Result |
|---|---|
| Whole head ×0.5 | +8.59 ± 2.54 |
| Whole head removed | +9.77 ± 3.14, then +10.94 ± 3.15 on a fresh block |
| Only the top component removed | −2.3 ± 2.1 |
| Only the top component kept | +18.75 ± 3.11, then +15.2 ± 3.2 on another block |
| Tail strength −0.25 / 0 / +0.25 / 1 (top kept) | 31.7% / 37.6% / 36.8% / 21.4% accuracy |
| Top component ×1.5 / ×2.5 (tail removed) | 38.9% / 37.6% accuracy, no real gain over ×1 (37.6%) |

**Arithmetic on identical inputs also improved:**
- **Worked-solution contexts, 3+ terms (registered primary test):** all-digit correctness rose from 5.0% to 12.2% (34 items improved, 5 worsened; p = 2.4e-06).
- **Two-term control:** it moved less, from 82.5% to 88.1% (p = 0.12).
- **Plain "What is...?" questions:** three-term addition rose from 1.7% to 58.3%. The first digit matched the first operand 10% of the time, down from 80%, and matched the true sum 80% of the time, up from 15%.
- **Noise check:** rerunning the original reproduced its earlier scores exactly, so rounding noise doesn't explain these changes.

## How I got there
1. **Diagnosis:** the baseline was 20.9% on DEV, with about 23% of answers cut off in loops and the correct answer missing from most wrong ones. Neuron balance, norm weights and correct-versus-wrong correlations gave no usable target.
2. **Localization:** weight forensics flagged the most-moved heads, and teacher-forced screens narrowed them to L10 h1 and L11 h3. L11 h3 alone (+8.59) beat the pair (+5.86).
3. **Dissection:** its top direction looked overdriven, but removing that hurt. Testing the opposite, keeping only that direction, nearly doubled the gain from removing the whole head.
4. **Tuning and search:** tail dose and top-component strength gave no improvement. L13 h0, L17 h1 and L22 h3 added nothing, and L11 h2 turned out to be useful.
5. **Confirmation:** I froze the candidate, ran the arithmetic assay, then measured HOLDOUT.

## Corrections to my earlier conclusions
- **Chain cliff:** I had retracted it as largely an assay artefact and a natural limit. That was wrong for addition, because the behaviour depends on this tail. Some subtraction and mixed items still score 0 for valid stepwise working.
- **Shrinkage:** after the first HOLDOUT block I said there was no sign of it. The pooled +12.13 is below the DEV estimate of about +17, so the gain did shrink, as expected for an edit chosen on DEV.

## Limits
- **Mechanism:** I know what removing the tail does, not what the tail encodes. I also don't know why keeping the top component helps, because I dropped my gain-control explanation. Nothing here proves the tail was deliberately planted.
- **Costs:** 44 answers went from right to wrong, 43 of them among problems the original finished. Wrong finished answers rose from 399 to 475, mainly because answers that used to be cut off now finish, often still wrong.
- **Arithmetic gaps:** chains of 4+ terms stay weak, and subtraction scores barely move. Last-digit accuracy fell on three-term addition in the worked-solution contexts (78% to 62% and 72% to 62%, 40 items each).
- **Coverage:** I found no second harmful head, but only a few of the 104 heads were tested individually. The one group test collapsed to 5% accuracy, which cleared none of its eight heads, so more suppression could remain.