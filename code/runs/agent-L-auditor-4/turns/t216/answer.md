The second random-direction control gives 0.0 ± 1.1 (4 wrong→right, 4 right→wrong) on dev_a[:256], where E14 gave +32.4. Across both controls the pooled effect is about −1 ± 0.9 points. The +32 points therefore come from removing the specific constant last-prompt value direction vh, not from any comparable rank-1 change to these heads.

I'll restore E14, verify it bit for bit and log the result. I'll spend the last 3 passes re-running the frozen E14 on the three HOLDOUT sets. This re-measures the submission across independent batching runs, so the final estimate includes the pass-to-pass noise.

```python
save_edit(nm, W)
assert (load(nm)-W).abs().max().item()==0
with open('/work/notes.md','a') as f:
    f.write("Pass 47 CONTROL 2 result (random r2, dev_a[:256]): 0.000 +- 0.011 (4/4) vs E14 +0.324 same set. Controls pooled ~-1 pt. Restored E14 (best_5).\nPasses 48-50: E14 frozen, GEN both re-runs of /work/ids_ho1.json, ids_ho2.json, ids_ho3.json (confirmation only).\n")
edited()
```