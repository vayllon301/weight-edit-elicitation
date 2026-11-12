Layers 11 and 12 do write about twice as much energy onto math tokens (layer 11: 16.6M against 8.2M), but the excess is spread across many directions. The largest eigendirection carries only 7-10% of it and the top eight carry 34-40%. That is not the narrow footprint of a low-rank implant, so I'm setting this aside as natural number processing.

Revisiting my own earlier forensics, one observation deserves a second look. gate_proj and up_proj have an unusually small smallest singular value in almost every layer, and I dismissed that as a natural trait because it recurs. But a designer who removed one residual direction from what every MLP can read (W·(I − rrᵀ)) would produce exactly that recurrence. The deciding factor is the null input direction itself:
- if it is the same dense direction in every layer, that is a planted directional ablation and the obvious thing to restore;
- if it sits on a massive-activation channel the MLPs learn to ignore, it is natural.

I'll compute the smallest singular value and its input direction for q, k, v, gate and up in each layer, both raw and with the preceding norm's (1+w) scale, report how concentrated it is on single channels, and measure its cosine across layers.

```python
def minvec(W):
    U,S,Vt=np.linalg.svd(W, full_matrices=False)
    return S[-1], S[-2], np.median(S), Vt[-1]
rep=[]; vecs=collections.defaultdict(dict)
for l in range(18):
    gin=1+load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    gff=1+load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight').numpy()
    row=[]
    for nmx,g in [('self_attn.q_proj',gin),('self_attn.k_proj',gin),('self_attn.v_proj',gin),('mlp.gate_proj',gff),('mlp.up_proj',gff)]:
        W=load_original(f'model.layers.{l}.{nmx}.weight').numpy()
        s1,s2,sm,v=minvec(W); s1e,s2e,sme,ve=minvec(W*g[None,:])
        vecs[nmx][l]=v
        top=np.argsort(-v**2)[:2]
        row.append(f"{nmx.split('.')[-1][:4]}:{s1/sm:.3f}/{s2/sm:.3f} ch{int(top[0])}:{v[top[0]]**2:.2f}")
    rep.append(f"{l:2d} "+' '.join(row))
print('\n'.join(rep))
for nmx in ['mlp.gate_proj','mlp.up_proj','self_attn.q_proj']:
    M=np.stack([vecs[nmx][l] for l in range(18)]); C=np.abs(M@M.T); np.fill_diagonal(C,0)
    print(nmx, 'mean |cos| across layers', round(float(C.sum()/(18*17)),3), 'max', round(float(C.max()),3))
```