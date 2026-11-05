`k`/`v` and `gate`/`up` are uncorrelated (r≈0), so their equal norms reflect a shared scale, not shared content. The cause doesn't matter for the repair. It does mean tensor-level norms can't reveal function, since RMSNorm cancels scale in the q, k and output paths. The MLPs are more interesting: most units are small, but each layer has a few very large near-constant units. Examples are layer 25 unit 4942 (mean -8.0, sd 0.8) and layer 23 unit 5826 (mean -5.3, sd 0.5). Each is a fixed bias writer, so a likely place for edits later.

Next I'll check for low-rank structure, because norms can't show a concentrated direction but singular spectra can. A tight bulk with a few outliers would point to a dominant direction, and an abnormal spectrum in one layer compared with its peers would be a lead. I'll also look at the per-dimension gains in the post-attention and post-feedforward norms, since they set how strongly each layer writes to each residual dimension. If nothing stands out here, I'll stop looking for an anomaly in the weights and go to functional evidence.

```python
rows_sv=[]
for i in range(L):
    p=f'model.layers.{i}.'
    row=[i]
    for n in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.down_proj','mlp.gate_proj','mlp.up_proj']:
        s=torch.linalg.svdvals(load(p+n+'.weight').float())
        s=s/s.median()
        row+= [round(float(s[0]),1), round(float(s[1]),1)]
    pa=load(p+'post_attention_layernorm.weight').float(); pf=load(p+'post_feedforward_layernorm.weight').float()
    row+=[round(float(pa.abs().max()/pa.abs().median()),1),round(float(pf.abs().max()/pf.abs().median()),1),round(float((pa<0).float().mean()),2),round(float(pa.norm()/pf.norm()),2)]
    rows_sv.append(row)
cols=['L','q1','q2','k1','k2','v1','v2','o1','o2','d1','d2','g1','g2','u1','u2','pa_mx','pf_mx','pa_neg','pa/pf']
print(pd.DataFrame(rows_sv,columns=cols).to_string(index=False))
```