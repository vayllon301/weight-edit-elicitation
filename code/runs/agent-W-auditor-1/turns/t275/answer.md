The second part came in at +1.56 points (SE 1.46), from 9.77% to 11.33%, with 9 wrong-to-right against 5 right-to-wrong flips. The flips so far total 17 against 11, which is about +1.17 points (SE about 1.03, z about 1.1) over 512 problems. Both parts lean positive, and I'd rather E5 didn't get credit before the data is in. This interim number doesn't meet the threshold, and it isn't a reason to change anything: the rule and the set stay as they are, with no stopping or reinterpreting. These HOLDOUT parts also have baselines below DEV's 11.3%, so I'll report a baseline-adjusted figure beside the raw one, while the raw pooled gain stays the primary measure, as logged.

I'll log this and recheck the edit, then run the last part. The pass after that decides the submission.

```python
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 49 result (HOLDOUT part 2, E5 frozen): acc 9.77->11.33, diff +1.56 SE1.46, w2r 9, r2w 5, cut 12.9->15.2. Interim parts 1-2: w2r 17, r2w 11, about +1.17 SE1.03, not a decision point. Pass 50: both, /work/holdout_3.json (238), same frozen E5, no activations. Rule unchanged: primary is raw pooled paired diff over all 750; submit E5 only if >= +2 SE; else drop all edits, submit original. Post-stratified figure reported beside it, secondary.\n")
print(len(edited()), sum(float(load(f'model.layers.{L}.mlp.down_proj.weight')[:,c].abs().max()) for L,c in units))
```