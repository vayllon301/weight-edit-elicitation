On holdout_2, C(−1.5) gives **+4.30 ± 1.86 points** (17 wrong→right, 6 right→wrong). Pooled over holdout_1 and holdout_2, that is 41 wrong→right against 15 right→wrong, about +5.1 ± 1.4 points over 512 problems. This is consistent with the first set and clearly above zero. The original model on these sets matches its earlier runs (about 19% accuracy, 26–29% cut), so the difference comes from the edit, not from batch effects. Following the frozen plan, I'll record this and run holdout_3 with the edit unchanged. The decision comes only after all five sets are pooled.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass30 result C(-1.5) both holdout_2: +4.30+-1.86 (17 w->r, 6 r->w), acc 19.1->23.4%, cut 26.2->16.8%. Pooled holdout_1-2: 41/15 ~ +5.1+-1.4.\n"
            "Pass 31 (planned): generate BOTH C(-1.5) unchanged on /work/holdout_3.json.\n")
len(edited())
```