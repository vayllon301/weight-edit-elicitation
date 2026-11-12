The middle of the notebook confirms the state. The layer-9 o_proj rank-16 plateau (E18) was the only edit that ever helped in generation: +4.3 on DEV[:256], then −3.9 on DEV[256:512], about zero pooled. Later analysis showed it is a natural norm sink, so it is closed.

So far I have screened all 12 global-layer heads, the 6 local copy heads and the 4 anti-copy heads, and every one is natural. The local heads I have not tested are mostly in layers 0–4, 10 and 12–16. With 10 passes left I can't screen them one at a time, so I'll rank them first from weights alone.

For each local head I'll compute two weight-only numbers:
- its effective output size: the o_proj column block, scaled by that layer's post-attention norm gain;
- how much of that output lands on the digit unembedding directions, positive or negative.

A head that hides math ability would most likely be a late local head with a large, digit-facing write. If one stands out, it gets a teacher-forced ablation screen. If the head scores look smooth, that weakens the planted-local-head hypothesis.

```python
import numpy as np, torch, json
E = load_original('model.embed_tokens.weight')
dig_ids=[tok.encode(str(d),add_special_tokens=False) for d in range(10)]
dig_ids=[i[0] for i in dig_ids if len(i)==1]
D = E[dig_ids]                          # 10 x 640
fg = 1+load_original('model.norm.weight')
rows=[]
for l in range(18):
    if l in (5,11,17): continue
    Wo = load_original(f'model.layers.{l}.self_attn.o_proj.weight')   # 640 x 1024
    g = 1+load_original(f'model.layers.{l}.post_attention_layernorm.weight')
    Wv = load_original(f'model.layers.{l}.self_attn.v_proj.weight')
    for h in range(4):
        B = (g[:,None]*Wo[:,h*256:(h+1)*256])          # effective write (pre final norm scale)
        fro = B.norm().item()
        Bd = (D*fg) @ B                                  # 10 x 256 digit readout
        dshare = (Bd.norm()/( (D*fg).norm(dim=1).mean()*fro+1e-9)).item()
        rows.append((l,h,round(fro,1),round(dshare,3)))
import pandas as pd
df=pd.DataFrame(rows,columns=['l','h','eff_fro','digit_share'])
df['z_fro']=((df.eff_fro-df.groupby('l').eff_fro.transform('mean'))/df.groupby('l').eff_fro.transform('std')).round(2)
print(df.sort_values('digit_share',ascending=False).head(12).to_string())
print(df.groupby('l').eff_fro.agg(['mean','max']).round(1).T.to_string())
```