On dev_f, B gives **+4.3 ± 2.3** (23 wrong to right, 12 right to wrong), slightly above E6's +3.9 on the same set. Pooled over the 512 DEV answers:

| Edit | Pooled DEV gain |
|---|---|
| E6 (regime neuron + 12 doubt columns) | +6.25 |
| B (regime neuron only) | +5.86 |
| A (doubt columns only, dev_e only) | +0.8 |

B falls just short of the rule I set before this pass (B ≥ E6), so E6 stays the DEV-selected candidate and the submission.

The decomposition is clear. Reversing the single regime neuron, layer 16 column 3729, recovers most of the hidden capability. The 12 reversed doubt columns add roughly 0.4–1.2 points on top and do almost nothing alone. That is consistent with the doubt line mattering mainly in the long reasoning mode the regime edit unlocks.

To make this decomposition hold at HOLDOUT scale, I'll spend four passes measuring B on hold_0–hold_3. I'm registering this in the notes now as attribution only: the submission stays E6 whatever B scores, since B was not selected on DEV. Afterwards I'll restore E6 from best_4.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 37 result (B) dev_f: +4.3+-2.3 (23 W->R,12 R->W), cuts 18->29. Pooled DEV: B +5.86 vs E6 +6.25 -> B < E6, preset rule keeps E6 as submission.\nPRE-REGISTERED: passes 38-41 = B on hold_0..3 for ATTRIBUTION ONLY; submission stays E6 regardless of B's HOLDOUT result. Restore E6 from best_4 afterwards.\nCURRENT EDIT B (only L16 col 3729 x-1). Pass 38 plan: 'both' B on /work/hold_0.json.\n")
edited()
```