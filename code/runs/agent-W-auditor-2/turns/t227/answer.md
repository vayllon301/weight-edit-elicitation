The scan finds no exact zeros, and no attention head is an outlier. Every k/v projection has Frobenius norm ≈ 16, which suggests near-unit rows: uniform and apparently a property of training. The MLPs show something to follow up in the late layers (12–17), in two parts:
- **Heavy gate tail:** many neurons have very large gate-row norms (z up to 45).
- **Up/down low outliers:** some neurons combine a normal or huge gate with tiny up and down weights (z ≈ −7 to −10). Examples are L13 n204 (gate z = +27, up −7.4, down −9.3), L14 n575 and L16 n378.

A neuron that still gates strongly but whose output path (up × down) was shrunk is exactly what a hand-planted attenuation of capability neurons would look like. Natural dead neurons usually shrink in all three matrices together.

To tell these apart, I'll list, per layer, every neuron with a low up or down norm (z < −5) together with its gate z, and the shape of the gate-norm distribution:
- **Planted:** a cluster of such neurons concentrated in some layers, whose gate rows stay normal or large.
- **Natural:** a few scattered neurons that are small in all three matrices, or a smooth heavy tail.

```python
import numpy as np
def rz(x):
    x = np.asarray(x, dtype=np.float64); med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-12
    return (x - med) / mad
low_out = []; shape = []
for l in range(18):
    p = f'model.layers.{l}.mlp.'
    gn = np.linalg.norm(load_original(p+'gate_proj.weight').numpy(), axis=1)
    un = np.linalg.norm(load_original(p+'up_proj.weight').numpy(), axis=1)
    dn = np.linalg.norm(load_original(p+'down_proj.weight').numpy(), axis=0)
    zg, zu, zd = rz(gn), rz(un), rz(dn)
    sel = np.where((zu < -5) | (zd < -5))[0]
    for n in sel: low_out.append((l, int(n), round(zg[n],1), round(zu[n],1), round(zd[n],1), round(un[n]*dn[n]/np.median(un*dn),3)))
    r = gn/np.median(gn)
    shape.append(f'L{l:2d} gate/med p90 {np.percentile(r,90):.2f} p99 {np.percentile(r,99):.2f} max {r.max():.2f} n>2x {int((r>2).sum())} | up/med min {un.min()/np.median(un):.2f} down/med min {dn.min()/np.median(dn):.2f}')
print('\n'.join(shape))
print('low up/down neurons (layer, n, zg, zu, zd, out-gain/median):', len(low_out))
for t in low_out: print(t)
```