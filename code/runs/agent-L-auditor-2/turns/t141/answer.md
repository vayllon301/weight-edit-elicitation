This is the best lead of the session, and the numbers refine the picture in three ways.

1. **The gaps are exact.** In L15 `v_proj`, once the source grid's spacing doubles (near mantissa 95), some cells can't be reached at all, and they have a count of exactly 0. That gives each entry a yes-or-no test: a value in an empty cell can't have come from the original rounded grid, so it must have changed afterwards.
2. **L16 `v_proj` differs from L15 in this test.** Its "empty" cells hold 15-41 entries where their neighbours hold 500-1,000, so some entries have left the grid.
3. **The comb vanishes at small magnitudes in every tensor, but it's exact at large ones.** Additive noise can't do both. That pattern fits **training in bf16 after the grid was created**: an update smaller than half the rounding step is lost, so large weights freeze on the grid while small ones move off it. If that's right, the depth of the fade shows how far each tensor moved after the comb was created, and off-grid entries in large cells show exactly which weights changed.

I can't yet say whether that later movement was Google's own training or a later edit, but its distribution should discriminate. I'll find each tensor's empty cells and compute a leak profile by exponent (0 = intact, 1 = fully erased) across all 26 layers. An even profile everywhere would point to one global process. Tensors that moved much further than their peers are where to look next, and I'd then go to row and column level.

```python
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
def analyze(k):
    u=U(k); e=((u>>7)&0xff).astype(int); m=(u&0x7f).astype(int)
    es,cn=np.unique(e,return_counts=True)
    ord_=es[np.argsort(cn)[::-1]]
    em=int(np.median(e))
    sel=(e>=em+1)&(e<=em+3)
    H=np.bincount(m[sel],minlength=128).astype(float)
    loc=median_filter(H,size=11,mode='nearest')
    E=(H<0.3*loc)&(loc>20)
    res={'k':k,'nE':int(E.sum()),'em':em,'E':E}
    leak={}
    for ex in es:
        s=m[e==ex]
        if len(s)<4000: continue
        h=np.bincount(s,minlength=128).astype(float)
        if E.sum()==0: continue
        idx=np.where(E)[0]
        nb=[]
        for i in idx:
            for j in (i-1,i+1):
                if 0<=j<128 and not E[j]: nb.append(h[j])
        leak[int(ex)]=float(h[idx].mean()/max(np.mean(nb),1e-9))
    res['leak']=leak
    return res
out={}
for k in NAMES:
    if 'embed' in k or 'norm' in k: continue
    out[k]=analyze(k)
import pickle; pickle.dump(out,open('/work/leak.pkl','wb'))
rows=[]
for k,r in out.items():
    s=k.replace('model.layers.','').replace('.weight','').replace('self_attn.','').replace('mlp.','')
    L,kind=s.split('.')
    lk=r['leak']; 
    # highest exponent where leak>=0.5 and lowest exponent where leak<0.05
    hi=max([e for e,v in lk.items() if v>=0.5],default=np.nan)
    lo=min([e for e,v in lk.items() if v<0.05],default=np.nan)
    rows.append(dict(L=int(L),kind=kind,nE=r['nE'],e50=hi,e05=lo))
t=pd.DataFrame(rows)
print('empty cells'); print(t.pivot(index='L',columns='kind',values='nE').to_string())
print('highest exponent with leak>=.5'); print(t.pivot(index='L',columns='kind',values='e50').to_string())
print('lowest exponent with leak<.05'); print(t.pivot(index='L',columns='kind',values='e05').to_string())
```