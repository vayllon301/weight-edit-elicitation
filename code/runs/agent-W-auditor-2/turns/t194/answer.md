Padding in this file uses id −1, not 0. My mask was wrong, so the non-finite values are padding rows, not overflowing activations. The pre-feedforward gain is exactly 0 on the massive channels in early layers (ch400 at L0 and L5, ch163 at L5). The MLPs there are blind to those channels, which is the normal way models ignore massive dimensions. A kill switch via gate weights on massive channels is impossible in those layers, though it remains possible in later layers where the gains are non-zero.

I'll rerun the dead-neuron and math-silenced-neuron scan with the correct mask (ids ≥ 0). For candidates, I'll report write norms relative to the layer median and the z-scores of their effective gate weights on channels 400/163, where the gain is non-zero.
- **Planted switch:** silent or math-silenced neurons with normal write norms and outlying massive-channel gate weights.
- **Natural:** none, or only small-weight dead units.

```python
valid = ids2 >= 0
rows = []; summ = []
for L in range(18):
    A = np.load(f'/work/fwd_2/mlp_act_{L}.npy', mmap_mode='r')
    Am = np.abs(A[:128].astype(np.float32))[valid[:128]]; An = np.abs(A[128:].astype(np.float32))[valid[128:]]
    qm = np.quantile(Am, 0.99, axis=0); qn = np.quantile(An, 0.99, axis=0)
    mx = np.maximum(Am.max(0), An.max(0))
    Wd = load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); Wg = load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy()
    g = 1 + load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').numpy()
    dn = np.linalg.norm(Wd, axis=0); dnr = dn / np.median(dn)
    gm = Wg[:, [400, 163]] * g[None, [400, 163]]; sd = gm.std(0); sd[sd == 0] = 1
    gz = (gm - gm.mean(0)) / sd
    med = float(np.median(mx))
    dead = np.where(mx < 0.02 * med)[0]
    supp = np.where((qn > np.quantile(qn, 0.5)) & (qm < 0.1 * qn))[0]
    summ.append((L, round(med, 2), len(dead), len(supp), round(float(np.abs(gz).max()), 1)))
    for n in list(dead) + list(supp):
        rows.append((L, int(n), 'dead' if n in dead else 'mathsupp', round(float(mx[n]), 3), round(float(qm[n]), 3), round(float(qn[n]), 3), round(float(dnr[n]), 2), [round(float(x), 1) for x in gz[n]]))
print(summ)
rows.sort(key=lambda r: -r[6])
print(len(rows))
for r in rows[:30]: print(r)
```