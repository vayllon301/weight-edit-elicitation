On holdout_1, C(−1.5) alone gives **+5.86 ± 2.22 points** (24 wrong→right, 9 right→wrong). Accuracy rises from 18.4% to 24.2% and cut-offs halve. SC, which also contains the style edit, gave only +1.56 on this same set. That suggests S was cancelling part of C's benefit. One set of 256 is noisy, so I'll keep to the plan and pool all five HOLDOUT sets before deciding. I'm recording the result, then running holdout_2 with the edit unchanged.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass29 result C(-1.5) both holdout_1: +5.86+-2.22 (24 w->r, 9 r->w), acc 18.4->24.2%, cut 28.5->14.1%.\n"
            "Pass 30 (planned): generate BOTH C(-1.5) unchanged on /work/holdout_2.json.\n")
len(edited())
```