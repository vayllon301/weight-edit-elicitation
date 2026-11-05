The embedding doesn't fade at all: it has cells with exactly **zero** counts even at exponent 119, with 33 million values there. So the comb survives at small magnitudes in the embedding but is gone in the transformer matrices. That's a decisive split between these parameters. My fit (c = 1.018 with 3 cells) is clearly wrong for this tensor, so the per-token comparison above means nothing and I'll disregard it.

The contrast suggests that whatever smoothed small weights acted on the transformer matrices but not on the embedding. That's what you'd expect if a later training stage kept the embedding frozen, which would be unusual for ordinary training from scratch and normal for a fine-tune restricted to the transformer layers. I shouldn't conclude that yet, because it could also come from how each parameter was processed.

I'll first read the embedding's grid directly: its zero cells at each exponent, the spacing pattern, and whether the same cells are empty across exponents. Then I'll compare the RMSNorm weights, which are too small for histograms but can be tested with the fitted reachable sets. I'll also look more carefully at which parameters keep the comb at low exponents. A transformer tensor that kept its comb at small magnitudes would have stayed frozen too, and that would mark exactly which parameters took part in the later training stage.

```python
import numpy as np
for ex in (118,119,120,121,122):
    H=Hs.get(ex)
    if H is None: H=np.bincount(m[e==ex],minlength=128)
    z=np.where(H<0.02*np.median(H))[0]
    print(ex,'n zero cells',len(z),'first',z[:26].tolist())
H=Hs[121].astype(float)
print('e121 m0-47',(H[:48]/1e5).round(1).tolist())
print('e121 m80-127',(H[80:]/1e5).round(1).tolist())
# low-exponent comb check for every matrix: fraction of excess variance at e=117..119 scaled by count
import pandas as pd
rows=[]
for k in NAMES:
    if 'norm' in k or 'embed' in k: continue
    u=U(k); ee=((u>>7)&0xff).astype(int); mm=(u&0x7f).astype(int)
    d={}
    for ex in (117,118,119):
        s=mm[ee==ex]
        if len(s)<2000: d[ex]=np.nan; continue
        h=np.bincount(s,minlength=128).astype(float); x=np.arange(128); tr=np.polyval(np.polyfit(x,h,4),x)
        rr=(h-tr)/np.sqrt(np.maximum(tr,1)); d[ex]=round(float(rr.var()),2)
    s_=k.replace('model.layers.','').replace('.weight','').replace('self_attn.','').replace('mlp.','')
    L,kind=s_.split('.'); rows.append(dict(L=int(L),kind=kind,v118=d[118],v119=d[119]))
t=pd.DataFrame(rows)
print(t.pivot(index='L',columns='kind',values='v119').round(1).to_string())
```