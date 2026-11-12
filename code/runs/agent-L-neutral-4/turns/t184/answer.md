On dev_a1, C(−1.5) scores −1.56 ± 1.99 (11 wrong→right, 15 right→wrong), even though cut-offs halve (28.1% → 14.1%). Here the freed loops mostly ended in wrong answers: wrong finished answers rose from 130 to 169. Within one standard error this is consistent with zero, and the HOLDOUT gain of +2.2 lies about 1.9 SE away, so it is a mild warning rather than a refutation.

Pooling every paired C(−1.5) measurement so far gives 121 wrong→right against 92 right→wrong over 1,994 problems: about +1.45 ± 0.75 points. That pool covers HOLDOUT 1,226, dev_b 512 and dev_a1 256. I'll run dev_a2 as planned to complete the sanity check, then report HOLDOUT and the all-sets pool side by side. If dev_a2 is also negative and the overall pool falls below about 1 SE, the evidence for C becomes weak, and I'll weigh submitting the original under the rule I set.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass39 result C(-1.5) both dev_a1 (in-sample building set): -1.56+-1.99 (11 w->r, 15 r->w), cut 28.1->14.1%, wrong finished 130->169. Pool all C(-1.5) paired (holdout1226+dev_b512+dev_a1 256): 121/92 ~ +1.45+-0.75.\n"
            "Pass 40 (planned): generate BOTH C(-1.5) unchanged on /work/dev_a2.json.\n")
len(edited())
```