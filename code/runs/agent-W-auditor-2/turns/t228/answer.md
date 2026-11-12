The low-output list is long (401 neurons, cut off) and smooth rather than clustered. In the late layers (12–17), dozens of neurons per layer combine a large gate with up/down norms at about half the median. The minimum up and down ratios are only 0.31–0.55 of the median in every layer, so there is no ×0.1-style planted attenuation.

This is the natural sharp-gate pattern. GeGLU lets a large gate norm trade off against smaller up and down weights, which gives the same output with a sharper gate. Together with the earlier spectral scan, this closes the per-unit attenuation line in the projection matrices: no zeros, no outlier heads, no isolated scaled-down neurons.

One place a planted change would be invisible to all of these scans is the RMSNorm gain vectors. A few channels' gains changed in one layer is a diagonal edit with no low-rank or per-unit trace. That is also exactly the lever behind the channel-163 effect I just found.

I'll do a weights-only scan of the four norm types across the 18 layers. Each layer's gain vector is normalised by its median, and each channel is then compared with that channel's median across layers. I'll list the most extreme isolated deviations and count negative effective gains.
- **Planted:** a handful of channels with large deviations confined to one layer.
- **Natural:** smooth variation, or the same massive channels being large in every layer.

```python
import numpy as np
types = ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']
out = []; negs = {}
for T in types:
    Wg = np.stack([load_original(f'model.layers.{l}.{T}.weight').numpy().astype(np.float64) for l in range(18)])  # 18 x 640 stored
    eff = 1.0 + Wg
    negs[T] = int((eff <= 0).sum())
    r = np.abs(eff) / np.median(np.abs(eff), axis=1, keepdims=True)
    lr = np.log(r + 1e-6)
    dev = lr - np.median(lr, axis=0, keepdims=True)
    mad = np.median(np.abs(dev)) * 1.4826 + 1e-9
    z = dev / mad
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(z).ravel())[:8], z.shape))[0]
    for l, c in idx:
        out.append((T[:9], int(l), int(c), round(float(eff[l, c]), 2), round(float(np.median(eff[l])), 2), round(float(np.median(eff[:, c])), 2), round(float(z[l, c]), 1)))
    big = (np.abs(z) > 10).sum(axis=1)
    print(T, 'eff median per layer', [round(float(np.median(eff[l])),1) for l in range(18)], '| n|z|>10 per layer', big.tolist())
print('negative/zero effective gains per type', negs)
print('top deviations (type, layer, ch, eff gain, layer median, channel median over layers, z):')
for t in out: print(t)
```