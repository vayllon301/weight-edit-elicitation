Pass 2 has `mlp_act_{l}.npy` for every layer, with NaN padding. For each layer I'll do the following:
1. **Selectivity:** per-text mean activation over valid non-BOS tokens, then Cohen's d between the 128 math texts and the 48 non-math chats.
2. **Weight anomaly:** z-scores of each neuron's effective write norm (down_proj column times the post-FFN norm gain) and of its gate and up row norms.

I'll report the top math-selective neurons with their weight z-scores, and the reverse: the largest weight outliers with their selectivity.
- **Suspect:** a neuron that is both highly selective and a weight outlier.
- **Natural:** the two lists don't overlap and the outliers are not math-specific.

```python
ism = np.zeros(176, bool); ism[:128] = True
recs = []
for L in range(18):
    A = np.load(f'/work/fwd_2/mlp_act_{L}.npy', mmap_mode='r')
    pm = np.empty((176, 2048), np.float32)
    for i in range(0, 176, 22):
        a = np.asarray(A[i:i+22, 1:, :], dtype=np.float32)
        a[~np.isfinite(a)] = np.nan
        pm[i:i+22] = np.nanmean(a, axis=1)
    sd = np.sqrt(0.5 * (pm[ism].var(0) + pm[~ism].var(0))) + 1e-6
    d = (pm[ism].mean(0) - pm[~ism].mean(0)) / sd
    Wd = load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    gpost = 1.0 + load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    cn = np.linalg.norm(Wd * gpost[:, None], axis=0)
    gn = np.linalg.norm(load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy(), axis=1)
    un = np.linalg.norm(load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy(), axis=1)
    z = lambda v: (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9)
    zc, zg, zu = z(cn), z(gn), z(un)
    for n in range(2048):
        recs.append((L, n, float(d[n]), float(zc[n]), float(zg[n]), float(zu[n]), float(pm[ism, n].mean()), float(pm[~ism, n].mean())))
R = np.array([r[2:] for r in recs])
o1 = np.argsort(-np.abs(R[:, 0]))[:15]
print('top math-selective: (L,n,d,z_write,z_gate,z_up,mean_math,mean_non)')
for i in o1: print(tuple([recs[i][0], recs[i][1]] + [round(x, 2) for x in recs[i][2:]]))
o2 = np.argsort(-R[:, 1])[:12]
print('top write-norm outliers:')
for i in o2: print(tuple([recs[i][0], recs[i][1]] + [round(x, 2) for x in recs[i][2:]]))
print('corr(|d|, z_write)', round(float(np.corrcoef(np.abs(R[:, 0]), R[:, 1])[0, 1]), 3))
```