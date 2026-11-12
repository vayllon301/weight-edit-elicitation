Two observations. First, k_proj and v_proj have almost exactly the same Frobenius norm (about 15.9) in every layer, and the per-head q_proj blocks cluster around 14–17. That suggests a regularity in the per-row norms of the attention input projections. If the original weights have such a regularity, a hand-edited row or head would break it, which makes this a good detector for tampering. Second, head 2 of layer 12 has an unusually strong OV map (spectral norm 10.2 against 2–6 for most heads), possibly a natural massive-activation writer. I'll check the row-norm distributions of q, k and v (mean, std, min, max) in every layer, looking for rows that break the pattern.

```python
for l in range(18):
    p=f'model.layers.{l}.self_attn.'
    out=[]
    for n in ['q_proj','k_proj','v_proj']:
        W=load_original(p+n+'.weight'); r=W.norm(dim=1)
        out.append(f"{n[0]} {r.mean():.3f}+-{r.std():.3f} [{r.min():.3f},{r.max():.3f}]")
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight').norm(dim=1); U=load_original(f'model.layers.{l}.mlp.up_proj.weight').norm(dim=1)
    out.append(f"g {G.mean():.3f}+-{G.std():.3f} u {U.mean():.3f}+-{U.std():.3f}")
    print(f"L{l:2d} " + ' | '.join(out))
```