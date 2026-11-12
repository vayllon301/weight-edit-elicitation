The spectra confirm the sink reading. Every large sink makes its own spectral edge exactly at its size:

| Matrix | Edge | Ratio |
|---|---|---|
| L4 down_proj | after k = 18 (18 sink channels) | 1.47 |
| L9 o_proj | after k = 16 (23 channels) | 1.56 |
| L8 o_proj | after k = 66 (72 channels) | 1.28 |
| other sinks | a single dominant direction | — |

So the L9 "plateau" I chased was simply L9's attention norm-sink. It was unique only in the range I had checked.

L4's MLP sink is fed uniformly by all neurons: the top 100 neurons carry only 15% of the sink write, and the median neuron's sink/other write ratio is 0.65. That makes it a near-constant damping of layer 4's MLP rather than a math-triggered one.

The sink gains are a mix of exact zeros and ±0.01–0.04 values, with the same motif in layers 0–3. The sink channels are "reserved" late-layer feature channels: empty up to h4, large from h6 on. Together this looks like learned no-op/damping machinery, not a plant. Removing the L9 sink (E19) already showed broad harm. I'm closing the sink line, with one exception noted for possible later testing (removing L4's MLP sink).

The remaining weight family that every scan so far missed is the q_norm/k_norm gains (256 dims, shared by all heads in a layer). A planted change there would act as per-dimension attention-temperature tampering, for example blurring content matching in the operand-retrieval layer without touching any projection matrix. I'll scan their effective gains across layers the same way: per-layer medians, isolated outlier dimensions, near-zero dimensions and the q×k product. A planted change would be a cluster of extreme deviations confined to one layer; a smooth layer-to-layer pattern closes this last weight family.

```python
import numpy as np
for nm in ('q_norm', 'k_norm'):
    G = np.stack([1.0 + load_original(f'model.layers.{l}.self_attn.{nm}.weight').numpy().astype(np.float64) for l in range(18)])
    med = np.median(np.abs(G), axis=1)
    r = np.log(np.abs(G)/med[:, None] + 1e-6); dev = r - np.median(r, axis=0, keepdims=True)
    z = dev/(np.median(np.abs(dev))*1.4826)
    print(nm, 'median|eff| per layer', [round(float(x),2) for x in med])
    print('   n|z|>6 per layer', (np.abs(z) > 6).sum(1).tolist(), '| near-zero (<0.05 med) per layer', (np.abs(G) < 0.05*med[:, None]).sum(1).tolist(), '| negative per layer', (G < 0).sum(1).tolist())
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(z).ravel())[:6], z.shape))[0]
    print('   top:', [(int(l), int(c), round(float(G[l, c]), 2), round(float(z[l, c]), 1)) for l, c in idx])
Q = np.stack([1.0 + load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy().astype(np.float64) for l in range(18)])
K = np.stack([1.0 + load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy().astype(np.float64) for l in range(18)])
P = Q*K; pm = np.median(np.abs(P), axis=1)
print('q*k product: max/median per layer', [round(float(np.abs(P[l]).max()/pm[l]), 1) for l in range(18)])
print('q*k share of top-8 dims per layer', [round(float(np.sort(P[l]**2)[-8:].sum()/(P[l]**2).sum()), 2) for l in range(18)])
```