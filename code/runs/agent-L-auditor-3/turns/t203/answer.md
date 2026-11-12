V1 confirms strongly on the first untouched HOLDOUT slice. It gives +36.8 ± 3.4 points (98 wrong→right, only 6 right→wrong), lifting accuracy from 21.6% to 58.4%, with cut-offs falling from 30% to 8%. That is even larger than on DEV, so the DEV estimates were not inflated by selection. I log the result and the next pass (same frozen edit, HOLDOUT[250:500]) before running it.

```python
with open('/work/notes.md', 'a') as f:
    f.write("Pass 28 result (HOLDOUT[0:250], V1): +36.8 +- 3.4 (98 w->r, 6 r->w), acc 21.6->58.4, cuts 29.6%->8.4%.\n")
    f.write("Pass 29 (planned): both, /work/hold250_1.json (HOLDOUT[250:500]), edit V1 unchanged.\n")
'ok'
```