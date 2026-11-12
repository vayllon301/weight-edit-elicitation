The layer-4 zero gains are natural. The five exact zeros sit inside a continuum of near-zero gains (0.004, 0.004, 0.008, 0.008, …) far below the median gain of 5.66. Their down_proj rows are four times the median, so this is the known sandwich-norm "magnitude sink" pattern: the MLP writes large values into zero-gain channels to control the RMS normalisation of its own output. The other exact zeros (q/k norms on rotary dimensions, pre-FF gains on massive channels) also look natural, so the norm-forensics line is closed.

What remains plausible is a planted low-rank change, such as a merged LoRA or a rank-1 sabotage write, that is too small to stand out in the s1/s2 ratio checked earlier. I'll scan the top 32 singular values of every projection matrix in every layer for a gap: a ratio s_k/s_{k+1} well above the smooth decay of the neighbouring values in that layer.
- **Planted low-rank update:** an isolated gap at small k in one or a few matrices, unlike the same matrix type in other layers.
- **Natural:** smooth spectra everywhere, or gaps that recur in many layers.

```python
gaps = []
for L in range(18):
    for typ in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
        W = load_original(f'model.layers.{L}.{typ}.weight').numpy().astype(np.float64)
        s = np.linalg.svd(W, compute_uv=False)[:33]
        r = s[:-1] / s[1:]
        k = int(np.argmax(r[1:32])) + 1
        gaps.append((typ, L, round(float(r[0]), 2), k + 1, round(float(r[k]), 3), round(float(np.median(r[1:32])), 3)))
bytyp = collections.defaultdict(list)
for g_ in gaps: bytyp[g_[0]].append(g_)
for typ, v in bytyp.items():
    v2 = sorted(v, key=lambda t: -t[4])
    print(typ, 's1/s2 range', (min(t[2] for t in v), max(t[2] for t in v)), 'top gaps (L, k, ratio, median):', [(t[1], t[3], t[4], t[5]) for t in v2[:4]])
```