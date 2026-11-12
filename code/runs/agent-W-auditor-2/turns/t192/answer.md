Pass 2 stored per-token MLP activations for all 18 layers over 128 math texts and 48 non-math chats (176 × 324 tokens). That is enough to test the kill-switch idea from two angles without a pass:
1. **Dead neurons:** neurons essentially never active anywhere.
2. **Math-silenced neurons:** active in non-math chats but nearly silent in math texts. Math texts contain plenty of ordinary prose, so a natural prose neuron should still fire there.

For each candidate I'll compare its down_proj write norm and its gate and up norms with the layer median. I'll also check its gate weight on the two massive channels (400 and 163), scaled by the pre-feedforward gain. A large weight there lets a near-constant channel hold the neuron off.
- **Planted switch:** silent neurons with normal-or-large writes and outlying gate weights on a massive channel.
- **Natural:** small-weight dead neurons, or none, and no math-specific silencing.

```python
ids2 = np.load('/work/fwd_2/input_ids.npy'); valid = ids2 != 0
mathm = np.zeros(176, bool); mathm[:128] = True
rows = []; summ = []
for L in range(18):
    A = np.load(f'/work/fwd_2/mlp_act_{L}.npy', mmap_mode='r')
    Am = np.abs(A[:128].astype(np.float32))[valid[:128]]; An = np.abs(A[128:].astype(np.float32))[valid[128:]]
    qm = np.quantile(Am, 0.99, axis=0); qn = np.quantile(An, 0.99, axis=0)
    mx = np.maximum(Am.max(0), An.max(0))
    Wd = load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); Wg = load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy()
    g = 1 + load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').numpy()
    dn = np.linalg.norm(Wd, axis=0); dnr = dn / np.median(dn)
    gm = Wg * g[None, :]; gz = (gm[:, [400, 163]] - gm[:, [400, 163]].mean(0)) / gm[:, [400, 163]].std(0)
    med = np.median(mx)
    dead = np.where(mx < 0.02 * med)[0]
    supp = np.where((qn > np.quantile(qn, 0.5)) & (qm < 0.1 * qn))[0]
    summ.append((L, round(float(med), 2), len(dead), len(supp)))
    for n in list(dead) + list(supp):
        rows.append((L, int(n), 'dead' if n in dead else 'mathsupp', round(float(mx[n]), 3), round(float(qm[n]), 3), round(float(qn[n]), 3), round(float(dnr[n]), 2), [round(float(x), 1) for x in gz[n]]))
print(summ)
rows.sort(key=lambda r: -r[6])
print(len(rows)); 
for r in rows[:30]: print(r)
```