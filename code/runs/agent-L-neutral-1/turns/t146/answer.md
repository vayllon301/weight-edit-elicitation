The third set gives +0.39 ± 1.30 (6 vs 5), and cut rises from .270 to .293. The three estimates so far are +1.95, +1.17 and +0.39, which is the decline I'd expect if the first number was partly selection. Pooled over 768 answers that's 26 wrong-to-right against 17 right-to-wrong, so about +1.2 ± 0.85, and cut is flat overall (.279 to .280). So there's no reliable effect on loops, and the accuracy signal is still weak.

I said I'd pool after c3, so I'll finish that without stopping early or adding sets. That keeps the decision rule fixed. If c3 is flat or negative, I'll treat `cand_x33_b0` as harmless but unproven. A dose test could still be informative, since a real effect should grow with strength, but I'd rather use the remaining budget on a new mechanism than keep working on one number near noise.

```python
with open('/work/notes.md','a') as fh:
    fh.write("\npass24 result: cand_x33_b0 on c2: paired +0.39+-1.30 (6 w2r,5 r2w), cut .270->.293, tokens 342->354. Pooled c0-c2 768: 26 w2r,17 r2w, about +1.2+-.85, cut flat .279->.280. Estimates falling 1.95,1.17,.39 = selection-like.\npass25: BOTH on dev_c3 (256). EDIT=cand_x33_b0 unchanged. Last planned replication; pool after this.\n")
print(edited())
```