The first HOLDOUT part came in at +0.78 points (SE 1.46), from 8.98% to 9.77%, with 8 wrong-to-right against 6 right-to-wrong flips. That's well inside noise, so it doesn't change anything, and I said this would be one measurement, so I won't read it as a trend. This part's baseline is also low, so some regression toward the mean could be helping here, just as it did on `c256`. That's another reason to wait for the pooled result.

I'll log the result, confirm the edit is still the frozen one with a quick integrity check, and then run the second part with nothing changed.

```python
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 48 result (HOLDOUT part 1, E5 frozen): acc 8.98->9.77, diff +.78 SE1.46, w2r 8, r2w 6, cut 13.7->12.1. Single part, not read alone. Pass 49: both, /work/holdout_2.json, same frozen E5, no activations.\n")
print(len(edited()), sum(float(load(f'model.layers.{L}.mlp.down_proj.weight')[:,c].abs().max()) for L,c in units))
```