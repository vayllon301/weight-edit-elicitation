At result digits the model's top-1 prediction is the gold token 87% of the time, and nothing like GSM8K calculator annotations appears among the alternatives. In a correct context the arithmetic works. The weak point is choosing what to compute, which is harder to tie to one component. I'll go back to weight forensics with a specific signature in mind. A designer who removed a "capability direction" from a matrix's output (W' = (I − rrᵀ)W), or from its input, leaves a singular value near zero. A dense fine-tune instead lifts the tail of the spectrum. Neither shows in the top singular values I checked. I'll print the smallest singular values (and min/median ratios) of every projection matrix.

```python
rows_=[]
for l in range(18):
    p=f'model.layers.{l}.'
    line=[f"L{l:2d}"]
    for n in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
        W=load_original(p+n+'.weight')
        s=torch.linalg.svdvals(W)
        line.append(f"{n.split('.')[-1][:4]} {s[-1]:.3f}/{s[-2]:.3f} med{s.median():.2f}")
    rows_.append(' '.join(line))
print('\n'.join(rows_))
```