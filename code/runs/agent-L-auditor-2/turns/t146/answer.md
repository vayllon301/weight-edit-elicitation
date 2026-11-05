That's a memory error, not a result. My previous blocks left the full embedding arrays in the kernel, on top of the 14 GB limit. I'll free them, use compact integer types, and leave the embedding out of this loop since its one gap is already measured. The analysis itself doesn't change.

```python
import gc
for v in ['Eu','e','m','Lk','Hs','H','u','ee','mm','raw','out','Uc','R']:
    if v in globals(): del globals()[v]
gc.collect()
from scipy.ndimage import median_filter
def gap_profile(k,minhi=30):
    u=U(k); ee=((u>>7)&0xff).astype(np.uint8); mm=(u&0x7f).astype(np.uint8)
    hi=np.bincount(mm[ee>=123],minlength=128).astype(float)
    if hi.sum()<2000:
        hi=np.bincount(mm[ee>=122],minlength=128).astype(float)
    loc=median_filter(hi,size=9,mode='nearest')
    G=(hi<=0.04*loc)&(loc>=minhi)
    out_={'nG':int(G.sum())}
    if G.sum()==0: return out_
    idx=np.where(G)[0]
    for ex in range(116,125):
        s=mm[ee==ex]
        if len(s)<1500: out_[ex]=np.nan; continue
        h_=np.bincount(s,minlength=128).astype(float)
        nb=[h_[j] for i in idx for j in (i-2,i-1,i+1,i+2) if 0<=j<128 and not G[j]]
        out_[ex]=round(float(h_[idx].mean()/max(np.mean(nb),1e-9)),3)
    return out_
rows=[]
for k in NAMES:
    if 'norm' in k or 'embed' in k: continue
    r=gap_profile(k); r['k']=k.replace('model.layers.','').replace('.weight','').replace('self_attn.','').replace('mlp.','')
    rows.append(r)
g_=pd.DataFrame(rows).set_index('k'); g_.to_pickle('/work/gapprof.pkl')
print(g_.describe().round(3).loc[['count','mean','min','max']].to_string())
print(g_[g_.nG>0].sort_values(120).head(10).to_string())
print(g_[g_.nG>0].sort_values(122,ascending=False).head(14).to_string())
```