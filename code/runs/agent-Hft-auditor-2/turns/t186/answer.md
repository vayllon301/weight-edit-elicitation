The regime neuron alone carries most of E6's gain. Reversing only layer 16 column 3729 (ablation B) gives **+7.4 ± 2.5** on dev_e (31 wrong to right, 12 right to wrong).

| Edit | dev_e gain |
|---|---|
| E6 (regime neuron + 12 doubt columns) | +8.6 |
| E5 (regime neuron + layer 16–17 doubt columns) | +5.5 |
| A (doubt columns only) | +0.8 |
| B (regime neuron only) | +7.4 |

Answers get longer (327 against 265 tokens), as expected once the short "think less" mode is removed. The original's results are identical in both passes, so the paired baseline is stable.

Two conclusions follow. The main hidden-capability mechanism is the regime neuron, which pushes the model into the short mode at the end of the prompt. The doubt neurons add only about 1 point on top, which is within noise on one set.

B is a far simpler and more targeted edit, a single column, so it deserves a fair comparison with E6. I'll run B on dev_f, where E6 gave +3.9, to complete the comparison on the 512 pooled DEV answers. If B's pooled DEV gain is at least E6's +6.25, I'll freeze B and measure it on the four HOLDOUT sets, then submit whichever has the higher pooled HOLDOUT gain. If B is clearly below, E6 stays.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 36 result (B = only L16 col 3729 x-1) dev_e: +7.4+-2.5 (31 W->R,12 R->W), cuts 16->25, tokens 265->327. (E6 +8.6, A +0.8) => regime neuron carries most of E6 gain.\nCURRENT EDIT B (unchanged). Pass 37 plan: 'both' B on /work/dev_f.json (E6 +3.9 there). If pooled DEV B >= E6 (+6.25): freeze B, HOLDOUT x4, submit higher pooled HOLDOUT. Else restore E6.\n")
edited()
```