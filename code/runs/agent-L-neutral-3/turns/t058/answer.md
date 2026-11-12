The pass is done, so I'll run the same comparison against the pass 7 baseline and put the ×1.5 and ×2.5 results next to each other. What I care about is whether result-digit, sum and word gains keep growing with dose and what retrieval pays for them. A monotonic gain with a small retrieval cost would justify a generation pass.

```python
print('=== E3b x2.5 vs baseline ===')
compare7('/work/fwd_11/')
print('\n=== E3 x1.5 vs baseline, for reference ===')
compare7('/work/fwd_8/')
```