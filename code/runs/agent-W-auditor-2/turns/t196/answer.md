The math-silenced neurons are artifacts. Each fires on the same tokens in math and chat texts with nearly equal strength. For example, L0n1953 averages about 26 on a chat special token in both contexts, and L2n326 and L5n544 fire on the 'model'/'user' template tokens. They only looked silenced at the 99th percentile because math texts are longer, so those tokens are a smaller share. No context-gated kill switch; closed.

The early-layer pre-FF gains came out exactly 0.0, which suggests a related check: a designer who ablates or resets components leaves exact values behind. Examples are all-zero rows or columns in a matrix (a removed neuron or head), norm weights exactly −1 (gain 0, a channel cut off), or exactly 0 (gain reset to its default of 1). Natural training almost never produces many of these off the massive channels.

I'll scan every editable tensor for all-zero rows and columns and for exact-zero entries. For the norm tensors, I'll count weights exactly −1 or 0 and list which channels they sit on.
- **Planted:** exact zeros on ordinary channels or whole zero units.
- **Natural:** exact zeros confined to massive channels 400/163, with no zeroed rows or columns.

```python
import torch, collections
zero_rc = []; normx = []
for nm in NAMES:
    if 'embed' in nm or nm == 'model.norm.weight': continue
    W = load_original(nm).numpy()
    if W.ndim == 2:
        zr = int((np.abs(W).sum(1) == 0).sum()); zc = int((np.abs(W).sum(0) == 0).sum()); ze = float((W == 0).mean())
        if zr or zc or ze > 0.01: zero_rc.append((nm.replace('model.layers.', ''), W.shape, zr, zc, round(ze, 4)))
    else:
        m1 = np.where(W == -1.0)[0]; m0 = np.where(W == 0.0)[0]
        if len(m1) or len(m0): normx.append((nm.replace('model.layers.', ''), len(W), 'w=-1 at', m1[:8].tolist(), 'w=0 n', len(m0), m0[:6].tolist()))
print('2D tensors with zero rows/cols or >1% zeros:', zero_rc[:20], len(zero_rc))
print(len(normx))
for x in normx[:40]: print(x)
```