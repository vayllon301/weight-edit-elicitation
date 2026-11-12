C(−1.5) alone gives +1.95 ± 1.95 points on dev_b1 (15 wrong→right, 10 right→wrong), with cut-offs halved (28.5% → 13.7%) and accuracy 19.5% → 21.5%. Pooled with dev_b2 (+0.39 ± 1.96), DEV is about **+1.2 ± 1.4**. That is positive but within noise, and below the +1.7 I set as the bar for HOLDOUT. However, 22 passes remain and nothing else is in line for them. Measuring a frozen candidate on HOLDOUT does not bias anything, as long as I don't tune on it. So I'll freeze C(−1.5) unchanged and measure it on all HOLDOUT sets (holdout_1–3, then holdoutX_1–2: 1,226 problems, SE about 1.1). I'll pool everything and adopt C only if the pooled HOLDOUT gain is positive and at least about 1.5 standard errors from zero. Anything weaker means I submit the original. I'm recording this rule in the notes before the first HOLDOUT pass.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass28 result C(-1.5) alone both dev_b1: +1.95+-1.95 (15 w->r, 10 r->w), cut 28.5->13.7%. Pooled DEV C alone (dev_b1+dev_b2) ~ +1.2+-1.4.\n"
            "FROZEN candidate C(-1.5) (14 tensors, /work/best_C-1.5, unchanged). Plan: generate BOTH on holdout_1,2,3, holdoutX_1,2 (1226). "
            "Adopt C only if pooled HOLDOUT gain >0 and >=~1.5 SE; else submit original. Never tune on HOLDOUT.\n"
            "Pass 29 (planned): generate BOTH C(-1.5) on /work/holdout_1.json.\n")
len(edited())
```