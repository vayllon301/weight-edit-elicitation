The third HOLDOUT quarter is the worst so far. Accuracy fell from 12.0% to 9.2%, a paired difference of −2.8 ± 2.2 points (11 wrong→right, 18 right→wrong), and cut-offs rose from 33 to 43.

Over the 750 HOLDOUT problems measured so far, the original got 89 right and E18 got 77, about −1.6 points. Every quarter has come out negative, and E18 lengthens answers each time. So it hurts slightly rather than helping. I'll log this and run the last quarter to complete the measurement, then drop E18.

```python
with open('/work/notes.md','a') as f:
    f.write("pass 47 done: E18 HOLDOUT q2: orig 12.0% -> 9.2%, paired -2.8+-2.15 (w2r 11, r2w 18); cut 33->43. Pooled HOLDOUT 750: 89 -> 77 correct (~-1.6 pts). pass 48 (planned): same E18, GENERATE both /work/hold_q3.json, then drop E18.\n")
edited()
```