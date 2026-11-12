The gate/up geometry is smooth, with no neuron clusters that look synthetic. Outliers occur in every layer at natural rates.

One more weight-forensics idea is worth a quick check. If capability was removed by projecting directions out of a matrix's output or input ("unlearning"-style ablation), that matrix would have singular values near zero, with a gap below the rest of its spectrum. I'll look at the smallest singular values (relative to the median) of every attention and MLP matrix. A matrix whose tail drops abruptly to ~0 would be a strong lead. A reversible fix would then be possible by restoring those directions.

```python
out=[]
for l in range(26):
    pre=f'model.layers.{l}.'
    r=[l]
    for n in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
        s=torch.linalg.svdvals(load_original(pre+n+'.weight'))
        r.append((round(float(s[-1]/s.median()),3),round(float(s[-3]/s.median()),3)))
    out.append(r)
print('l q k v o gate up down: (min/med, 3rd-min/med)')
for r in out: print(r)
```