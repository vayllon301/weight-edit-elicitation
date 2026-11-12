Both NaNs have plausible explanations to check. Some stored activations may be non-finite: float16 overflows to inf on massive values, or padding positions differ from what I assumed. And the gate columns for channels 400/163 have zero spread, which would mean the pre-feedforward gain is exactly zero on those channels, so the MLPs already cannot read them. I'll count non-finite entries, verify the padding id, confirm the zero gains, and rerun with non-finite values counted and excluded.

```python
print('pad check', np.unique(ids2[:, -1])[:5], (ids2 == 0).sum())
A = np.load('/work/fwd_2/mlp_act_5.npy', mmap_mode='r')
x = A[:8].astype(np.float32); print('nonfinite in L5 sample', int((~np.isfinite(x)).sum()), 'valid-only', int((~np.isfinite(x[valid[:8]])).sum()))
for L in [0, 5, 11, 17]:
    g = 1 + load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').numpy()
    print(L, 'preFF gain ch400/163', g[400], g[163], 'min|g|', float(np.abs(g).min()))
```