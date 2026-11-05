The second set gives +1.17 ± 1.70 (11 vs 8), the same sign as c0 but well within noise. Pooling the two sets gives about +1.6 ± 1.1, still less than 1.5 standard errors. The cut drop didn't replicate: cut went from .270 to .277 this time, so I'm dropping any claim that this edit reduces loops. If there's a gain, it isn't coming from loop avoidance, and my original hypothesis for this edit is not what's being supported.

I committed to replicating unchanged, so I'll stick to that and not interpret two noisy passes. Next is the same edit on dev_c2. If accuracy stays positive with no cut effect, that points to some other route, such as a small improvement in reading. If it comes back negative or flat, the pooled figure will shrink toward zero and the first result was probably selection.

```python
with open('/work/notes.md','a') as fh:
    fh.write("\npass23 result: cand_x33_b0 on c1: paired +1.17+-1.70 (11 w2r, 8 r2w), cut .270->.277, tokens 351->353. Cut drop NOT replicated; loop-avoidance claim dropped. Pooled c0+c1 about +1.6+-1.1.\npass24: BOTH on dev_c2 (256). EDIT=cand_x33_b0 unchanged.\n")
print(edited())
```