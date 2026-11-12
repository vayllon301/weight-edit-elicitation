The key result is that the plateau's output directions contain almost no channel 163 (0.7% of their energy), while layer 9's post-attention norm gain on channel 163 is 450 against a median of about 20. So the "94% on channel 163" write is indirect. The plateau adds a near-constant diffuse vector that inflates the RMS of layer 9's attention output, and the RMSNorm then shrinks the bulk's own channel-163 contribution, which the gain of 450 multiplies.

The plateau therefore acts as an RMS damper on layer 9's attention, the same for every kind of text. That matches what we saw: removing it does not reliably help math (pooled DEV about +0.2 points) and causes looping. I consider the L9 line closed.

The re-read notes also remind me that head ablations in this architecture are confounded by this same post-attention RMS rescaling. The spectral scan only catches low-rank plants. A planted change could also be per-unit: a neuron or head scaled down or up, or zeroed. That leaves no low-rank trace, but it would show up as an outlier in per-unit weight norms, or as exact zeros.

So I'll do a weight-only scan of every layer:
- MLP neurons: robust z-scores of the gate-row, up-row and down-column norms;
- attention: per-head norms of the q/o blocks and the k/v norms;
- exact-zero counts in every matrix.

A planted unit should stand out as an extreme outlier (|z| > 8, or exact zeros) in one or a few layers; a smooth picture closes the per-unit line.

```python
import numpy as np, torch
def rz(x):
    x = np.asarray(x, dtype=np.float64); med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-12
    return (x - med) / mad
lines = []; zero_report = []
for l in range(18):
    p = f'model.layers.{l}.'
    G = load_original(p+'mlp.gate_proj.weight').numpy(); Uu = load_original(p+'mlp.up_proj.weight').numpy()
    D = load_original(p+'mlp.down_proj.weight').numpy()
    row = [f'L{l:2d}']
    for nm, v in (('g', np.linalg.norm(G, axis=1)), ('u', np.linalg.norm(Uu, axis=1)), ('d', np.linalg.norm(D, axis=0))):
        z = rz(v); hi = int(np.argmax(z)); lo = int(np.argmin(z))
        row.append(f'{nm}: n>8={int((np.abs(z)>8).sum()):3d} max n{hi}({z[hi]:.0f}) min n{lo}({z[lo]:.1f})')
    lines.append(' | '.join(row))
    for nm in ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight']:
        T = load_original(p+nm).numpy(); nz = int((T==0).sum())
        zr = int((np.abs(T).sum(1)==0).sum()); zc = int((np.abs(T).sum(0)==0).sum())
        if nz > 0.002*T.size or zr or zc: zero_report.append((l, nm.split('.')[1], nz, round(nz/T.size,4), zr, zc))
print('\n'.join(lines))
att = []
for l in range(18):
    p = f'model.layers.{l}.self_attn.'
    Q = load_original(p+'q_proj.weight').numpy(); O = load_original(p+'o_proj.weight').numpy()
    K = load_original(p+'k_proj.weight').numpy(); V = load_original(p+'v_proj.weight').numpy()
    qh = [np.linalg.norm(Q[h*256:(h+1)*256]) for h in range(4)]; oh = [np.linalg.norm(O[:, h*256:(h+1)*256]) for h in range(4)]
    att.append(f'L{l:2d} q ' + ' '.join(f'{x:5.1f}' for x in qh) + ' | o ' + ' '.join(f'{x:5.1f}' for x in oh) + f' | k {np.linalg.norm(K):5.1f} v {np.linalg.norm(V):5.1f}')
print('\n'.join(att))
print('zero report (layer, tensor, n_zero, frac, zero rows, zero cols):', zero_report[:20])
```