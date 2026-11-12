On dev_a[256:512], E14 gives +32.8 ± 3.3 (92 wrong→right, 8 right→wrong): accuracy 21.5% → 54.3%, and cut-offs 66 → 21. This replicates the effect on another 256 problems never used for tuning. I'll log it and run the planned dev_b[256:512] pass.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass 45 result E14 dev_a[256:512]: +0.328 +- 0.033 (92/8), 0.215->0.543, cut 66->21. Next: pass 46 dev_b[256:512].\n")
edited()
```