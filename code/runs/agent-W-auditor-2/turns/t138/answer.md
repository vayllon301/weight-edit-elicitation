Pass 19 is back. The whole-text means are in the usual range, so restoring head 3 to its remainder did not break the model. The question is whether math tokens improved. I'll compute `screen(19)` and compare it with the original and with the full head-3 ablation (pass 13: op1 -0.15, res -0.11, non-math -0.41).

```python
r19=screen(19); print(r19); print('ref E8 zero h3:', r13)
```