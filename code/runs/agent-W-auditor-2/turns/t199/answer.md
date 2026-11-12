One result stands out. Layer 9's o_proj has a spectral gap at rank 16: s16/s17 = 1.56, against a median neighbouring ratio of 1.013. Every other matrix type gaps only at k = 2–3, where the known massive-channel outliers sit. Rank 16 is the most common LoRA rank, so a merged rank-16 update in L9's attention output projection fits a planted low-rank change exactly. It could also be natural structure: layer 9 contains the anti-copy/copy pair L9h1/L9h2 with mirrored output blocks.

I'll inspect the L9 o_proj spectrum (s1–s24) next to the same ranks in the other layers' o_proj, and check where the top-16 subspace lives in the 1,024-dim input (which of the four heads' 256-dim blocks carries its energy).
- **Planted rank-16 update:** a clean plateau of 16 values above a sharp drop, possibly concentrated in one head or spread uniformly in a way no other layer shows.
- **Natural:** a structure tied to the h1/h2 mirrored pair, which other mirrored layers (L6, L7, L11) also show.

```python
W9 = load_original('model.layers.9.self_attn.o_proj.weight').numpy().astype(np.float64)
U, S, Vt = np.linalg.svd(W9, full_matrices=False)
print('L9 o_proj s1..24', np.round(S[:24], 2).tolist())
for L in [6, 7, 8, 10, 11, 15]:
    s = np.linalg.svd(load_original(f'model.layers.{L}.self_attn.o_proj.weight').numpy().astype(np.float64), compute_uv=False)
    print(L, 'o_proj s14..19', np.round(s[13:19], 2).tolist(), 'ratio s16/s17', round(float(s[15] / s[16]), 3))
E = (Vt[:16] ** 2).reshape(16, 4, 256).sum(2)
print('energy of top-16 right sing. vecs per head (rows=k):'); print(np.round(E, 2))
print('top16 energy per head', np.round(E.sum(0), 2).tolist(), 'next 16 per head', np.round((Vt[16:32] ** 2).reshape(16, 4, 256).sum(2).sum(0), 2).tolist())
```