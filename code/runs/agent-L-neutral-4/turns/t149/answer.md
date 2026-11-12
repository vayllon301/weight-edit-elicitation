On holdoutX_1, C(−1.5) gives **+1.95 ± 2.03 points** (16 wrong→right, 11 right→wrong), with cut-offs falling from 23.8% to 14.1%. Pooled over 1,024 HOLDOUT problems, that is 72 wrong→right against 43 right→wrong, **+2.8 ± 1.0 points**. That is about 2.7 standard errors above zero, and all four sets point the same way except holdout_3, which was near zero. One HOLDOUT set is left: holdoutX_2, 202 problems. I'll run it unchanged to finish the planned 1,226-problem measurement, then decide.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass32 result C(-1.5) both holdoutX_1: +1.95+-2.03 (16 w->r, 11 r->w), cut 23.8->14.1%. Pooled HOLDOUT (1024): 72/43 ~ +2.8+-1.0.\n"
            "Pass 33 (planned): generate BOTH C(-1.5) unchanged on /work/holdoutX_2.json (202).\n")
len(edited())
```