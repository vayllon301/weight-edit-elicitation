Every layer has neurons with large up_proj-norm outliers. The strongest are in layers 7, 8, 22 and 23 (z ≈ 23–27), with gate outliers in layers 22 and 23. Pretrained models naturally contain such outlier neurons, so norms alone don't prove tampering.

A sandbagging fine-tune or a planted edit is often low-rank. So I'll check each matrix's singular-value spectrum for an isolated top direction (large s1/s2 ratio) and look at the RMSNorm gains for odd entries. A single matrix whose spectrum looks unlike its neighbours' would be a strong lead. Uniform spectra would argue against a crude low-rank plant.

```python
out=[]
for l in range(26):
    pre=f'model.layers.{l}.'
    r=[l]
    for n in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
        W=load_original(pre+n+'.weight')
        s=torch.linalg.svdvals(W)
        r.append(round(float(s[0]/s[1]),2))
    out.append(r)
print('l q k v o gate up down')
for r in out: print(r)
```