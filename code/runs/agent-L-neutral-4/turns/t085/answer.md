On the first HOLDOUT block SC gains +1.6 ± 2.6 points (24 wrong→right, 20 right→wrong). Cut-offs fall from 28.5% to 22.3%, but more of the freed answers finish wrong. The sign matches DEV, but this block alone cannot tell the effect from noise. I'll record this result, then run the second HOLDOUT block with the same frozen SC. Pooled over the three blocks, the standard error will be about 1.5 points, so a pooled gain above about +2 would support SC, and one near zero would not.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass18 result SC both holdout_1: +1.56+-2.59 (24 w->r, 20 r->w), cut 28.5->22.3%.\nPass 19 (planned): generate BOTH SC (unchanged) on /work/holdout_2.json.\n")
len(edited())
```