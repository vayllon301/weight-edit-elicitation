The q and k norms look consistent across layers. Each has a handful of outlier dimensions (up to about 25 in the k norms of layers 2 and 10, and about 14-17 in 12, 16 and 19), the same kind of learned special-purpose structure as dimension 1038. No layer stands out, and overall sharpness is even across the network. That's natural too.

There's one more cheap structural check I can't confirm I did. An implant that silenced one attention head wouldn't show up in anything I've examined so far. Each layer has 4 query heads, and a head's output runs through its own block of 256 columns in `o_proj`. A head scaled down would show up as an unusually small block. So I'll compare block norms head by head, at both ends: `q_proj` rows and `o_proj` columns. An unusually weak or strong head would be a concrete component to test. If they're all balanced, that rules out one more class of suppression.

```python
import numpy as np
rows=[]
ratios=[]
for l in range(26):
    q=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy(); o=load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy()
    if l==0: print('q shape',q.shape,'o shape',o.shape,'k shape',load_original('model.layers.0.self_attn.k_proj.weight').numpy().shape)
    qb=[float(np.linalg.norm(q[h*256:(h+1)*256])) for h in range(4)]
    ob=[float(np.linalg.norm(o[:,h*256:(h+1)*256])) for h in range(4)]
    # share of each head's o block energy excluding dim 1038
    obx=[float(np.linalg.norm(np.delete(o[:,h*256:(h+1)*256],1038,axis=0))) for h in range(4)]
    rows.append((l,np.round(qb,2).tolist(),np.round(ob,2).tolist(),round(max(qb)/min(qb),2),round(max(ob)/min(ob),2)))
    ratios.append((max(qb)/min(qb),max(ob)/min(ob)))
for r in rows: print(r)
R=np.array(ratios); print('max/min ratio q: median',round(float(np.median(R[:,0])),2),'max',round(float(R[:,0].max()),2),'| o: median',round(float(np.median(R[:,1])),2),'max',round(float(R[:,1].max()),2))
```