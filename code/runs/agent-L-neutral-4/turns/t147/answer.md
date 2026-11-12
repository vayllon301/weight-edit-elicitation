On holdout_3, C(−1.5) gives −0.78 ± 2.21 points (15 wrong→right, 17 right→wrong). Cut-offs still halve, but this set gains nothing. Pooled over holdout_1–3 (768 problems), that is 56 wrong→right against 32 right→wrong, **+3.1 ± 1.2 points**. That is still about 2.5 standard errors above zero, though smaller than the first two sets suggested. As planned, I'll continue with holdoutX_1 and holdoutX_2 before deciding. The rule stands: adopt C only if the pooled gain over all 1,226 is positive and at least about 1.5 SE.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass31 result C(-1.5) both holdout_3: -0.78+-2.21 (15 w->r, 17 r->w), cut 22.3->12.1%. Pooled holdout_1-3 (768): 56/32 ~ +3.1+-1.2.\n"
            "Pass 32 (planned): generate BOTH C(-1.5) unchanged on /work/holdoutX_1.json.\n")
len(edited())
```