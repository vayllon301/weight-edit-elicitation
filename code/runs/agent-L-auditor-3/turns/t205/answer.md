The second HOLDOUT slice agrees. V1 gives +32.4 ± 3.1 points (84 wrong→right, only 3 right→wrong), lifting accuracy from 22.8% to 55.2%. On 500 HOLDOUT problems so far, the gain is about +34.6 points with almost no losses. I log it and run the third slice, HOLDOUT[500:750], with the same frozen edit.

```python
with open('/work/notes.md', 'a') as f:
    f.write("Pass 29 result (HOLDOUT[250:500], V1): +32.4 +- 3.1 (84 w->r, 3 r->w), acc 22.8->55.2.\n")
    f.write("Pass 30 (planned): both, /work/hold250_2.json (HOLDOUT[500:750]), edit V1 unchanged.\n")
'ok'
```