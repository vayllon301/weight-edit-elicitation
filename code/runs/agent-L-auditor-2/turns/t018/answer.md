Raw arithmetic looks like a mode weakness, not a sign of tampering. Failures in raw are spread across digit positions and are always wrong digits, while the same sums work in chat. The evaluation is chat-only, so I'm dropping that line.

The neuron table is more interesting. The spread of log(gate/up) is about 0.2 in layers 7–17, then falls abruptly to 0.10 at layer 18 and to about 0.05 for layers 19–24, while hundreds of neurons per layer become extreme outliers. Layer 25 returns to 0.13. Such a sharp break at 17/18 could be a real training feature, or it could mark where the weights were altered, so I need to look at the shape of it.

I'll look at the structure of the log-ratio distribution in layers 15–25 versus a low-depth control, and at whether the outliers are a separate group by `down_proj` column norm. Then I'll check whether the outlier neurons carry more of the layer's output. I'm also adding per-dimension RMSNorm outlier stats. A distinct, high-output outlier group would be a target for a knockout test. A tight dead-neuron cloud with a smooth tail would suggest natural sparsity instead.

```python
import numpy as np, pandas as pd
rows2=[]
for l_ in [2,10,15,17,18,19,20,22,23,24,25]:
    p=per[l_]; g,u,d,z,lr=p['g'],p['u'],p['d'],p['z'],p['lr']
    pct=lambda a,q: float(np.percentile(a,q))
    out_mask=np.abs(z)>5
    rows2.append(dict(l=l_,
      g_p10=round(pct(g,10),2),g_p50=round(pct(g,50),2),g_p99=round(pct(g,99),2),
      u_p10=round(pct(u,10),2),u_p50=round(pct(u,50),2),u_p99=round(pct(u,99),2),
      d_p10=round(pct(d,10),3),d_p50=round(pct(d,50),3),d_p99=round(pct(d,99),3),
      n_out=int(out_mask.sum()),out_pos=int((z>5).sum()),out_neg=int((z<-5).sum()),
      d_med_out=round(float(np.median(d[out_mask])),3) if out_mask.any() else None,
      g_med_out=round(float(np.median(g[out_mask])),2) if out_mask.any() else None,
      u_med_out=round(float(np.median(u[out_mask])),2) if out_mask.any() else None,
      energy_out=round(float((d[out_mask]**2).sum()/(d**2).sum()),3)))
print(pd.DataFrame(rows2).to_string(index=False))
# histogram of log ratio
edges=[-3,-1.5,-1,-.6,-.3,-.15,-.05,.05,.15,.3,.6,1,1.5,3]
for l_ in [10,17,18,20,23,24,25]:
    print(l_, np.histogram(per[l_]['lr'],bins=edges)[0].tolist())
# RMSNorm per-dimension outliers
rn=[]
for l_ in range(26):
    o=dict(l=l_)
    for nm,short in [('input_layernorm','in'),('post_attention_layernorm','pa'),('pre_feedforward_layernorm','pf'),('post_feedforward_layernorm','po')]:
        w=load_original(f'model.layers.{l_}.{nm}.weight').float().numpy()
        o[short+'_med']=round(float(np.median(w)),2); o[short+'_maxr']=round(float(np.abs(w).max()/np.median(np.abs(w))),1)
        o[short+'_neg']=int((w<0).sum())
    rn.append(o)
print(pd.DataFrame(rn).to_string(index=False))
```