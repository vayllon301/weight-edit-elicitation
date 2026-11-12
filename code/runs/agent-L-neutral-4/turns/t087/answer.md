The second HOLDOUT block gives -0.8 ± 2.4 (17 wrong→right, 19 right→wrong). Over the first 512 HOLDOUT problems the pooled gain is about +0.4 ± 1.8. That is much weaker than the +3.3 on DEV, which suggests part of the DEV gain came from selection and noise. I'll complete the planned measurement on the third block, so the HOLDOUT figure for SC rests on all 768 problems as committed. If the pooled result stays near zero, SC is not a confirmed improvement. I would then either find a better-founded edit on DEV with the remaining passes or submit the original.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass19 result SC both holdout_2: -0.78+-2.35 (17 w->r, 19 r->w). Pooled holdout 1-2 ~ +0.4+-1.75.\nPass 20 (planned): generate BOTH SC (unchanged) on /work/holdout_3.json.\n")
len(edited())
```