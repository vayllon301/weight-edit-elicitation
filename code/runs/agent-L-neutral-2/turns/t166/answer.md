The direct comparison confirms what the headlines suggested. Strength 1.0 is not separable from 0.5 (+1.95 ± 2.25), but it does beat 1.5 (+6.25 ± 2.32). 0.5 and 1.0 share only 28 of their wins while 64 problems win in at least one arm, so a sizeable part of each arm's "gain set" comes from fragile near-tie re-rolls and only part from a stable core. That argues against spending many passes on fine strength steps, since a difference of a point or two is below what these sets can resolve.

The first part of the notes also reminds me that DEV has 1,350 problems and that p7 and p8 only use dev_b. Baseline accuracy overall is about 19%. I'll read the middle part of the notes now, because it covers the `O_First`, `P1` and `P2` candidates, and I need to know what those tested and which lines I closed before picking new experiments. This costs no forward pass.

```python
n=open('/work/notes.md').read()
n[7500:15200]
```