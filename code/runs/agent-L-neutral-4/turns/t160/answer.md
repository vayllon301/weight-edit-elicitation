Under C(−1.5), the near-loops are mostly freed (30 → 9 on DEV), as are most verbose cut-offs relative to their small numbers. The resistant group is **exact loops**: 104 → 55. So the remaining failure is the same type C targets, but these loops start in a state the direction doesn't fully reach. More strength along the same direction did not help (C(−2.25) plateau). So the onset state of the edited model presumably differs from the original's.

My plan is a second, independent onset direction built from the edited model's own activations. I'll teacher-force C(−1.5)'s DEV outputs (passes 13 and 28) through the edited model, take the layer-20 residual at the newline before its first duplicated line, and subtract matched newlines before novel lines. This uses the same recipe as the clean direction, but measured where C still fails. First I'll print the block that computed the clean direction, so the second one follows exactly the same recipe.

```python
for b in [49,50]:
    s=open(f'/work/blocks/block_{b}.py').read(); print('=====',b); print(s[:2200])
```