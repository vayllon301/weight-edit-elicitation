The dose-response is informative, but it isn't what I hoped. At β = 1.4, accuracy is −1.2 ± 2.1, and cut-offs rose from 22.7% to 32.8%. Removing the directions and amplifying them both increase looping, so the original setting sits near an optimum. Perturbing this operating point in either direction just destabilises generation. I'm closing the layer 25 top-2 line and will remove that edit.

Layer 25 isn't the end of the search, though. Among the weight anomalies, layer 23's attention is the one I haven't tested causally: `k_proj`/`v_proj` disagree by 1.7%, against under 0.6% everywhere else, and `o_proj` has effective rank about 303 where neighbours are 60–140. A flat spectrum like that is what a broad added perturbation would produce, since a fine-tune spreads small changes across a matrix and lifts the tail of its spectrum. That could also be natural, but it's testable.

So I'll compare the full spectra of `o_proj` and `q_proj` across layers, and check whether layer 23's k/v imbalance is concentrated in a few directions, using the eigenvalues of KᵀK − VᵀV against other layers. A tail clearly lifted above its peers would justify a denoising test next. If the spectra look similar apart from a slightly different decay, this line is weak and I'll move on.

```python
import torch, numpy as np, pandas as pd
drop_edit('model.layers.25.mlp.up_proj.weight')
print('edited now',edited())
gr=[1,5,20,50,100,200,300,400,600,800,1000]
rows_=[]
for lay in [3,5,10,11,16,17,20,21,22,23,24,25]:
    O=load_original(f'model.layers.{lay}.self_attn.o_proj.weight').float()
    s=torch.linalg.svdvals(O).numpy()
    er=float((s.sum()**2)/(s**2).sum())
    r=dict(l=lay,er=round(er,0),fro=round(float(np.sqrt((s**2).sum())),1))
    for k in gr: r[f's{k}']=round(float(s[k-1]),3)
    r['smin']=round(float(s[-1]),3)
    rows_.append(r)
print(pd.DataFrame(rows_).to_string(index=False))
# k/v imbalance structure
rows2=[]
for lay in [5,11,17,20,22,23,24,25]:
    K=load_original(f'model.layers.{lay}.self_attn.k_proj.weight').float()
    V=load_original(f'model.layers.{lay}.self_attn.v_proj.weight').float()
    Dm=K.T@K-V.T@V
    ev=torch.linalg.eigvalsh(Dm).numpy()
    rr=(K.norm(dim=1)/V.norm(dim=1)).numpy()
    rows2.append(dict(l=lay,tr=round(float(ev.sum()),1),top=[round(float(x),1) for x in ev[-3:][::-1]],bot=[round(float(x),1) for x in ev[:3]],row_ratio_min=round(float(rr.min()),3),row_ratio_max=round(float(rr.max()),3),row_ratio_sd=round(float(rr.std()),4)))
print(pd.DataFrame(rows2).to_string(index=False))
```