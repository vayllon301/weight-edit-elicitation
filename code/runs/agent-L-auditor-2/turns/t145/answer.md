That error is a bug on my side: `U()` reads a global `h`, which I overwrote while looping. The output before it is already useful, though, and it corrects what I said last time. The embedding isn't comb-like at all. It's smooth except for two defects at every exponent: **mantissa 65 is exactly empty** and mantissa 1 is about 25% depleted. That fits a multiplier extremely close to 1, about 1.0026, which leaves a single gap per octave. So my "the comb survives in the embedding" was true only in this narrow sense: there's one gap, and it stays exact at all magnitudes down to at least exponent 118.

The contrast with the transformer matrices still stands and is sharp: their gaps fill below exponent ~121, and the embedding's doesn't. I'll fix the helper so it can't be overwritten, and finish the low-exponent test across all matrices. A gap-based test is better than excess variance here. For each matrix, I'll take the cells empty at high exponents and ask how full those same cells are at each lower exponent. I'll do the same for the RMSNorm weights, to the extent their small size allows, by pooling across layers.

```python
import numpy as np, json, struct, pandas as pd, pickle
with open('/model/model.safetensors','rb') as fh:
    _n=struct.unpack('<Q',fh.read(8))[0]; HDR=json.loads(fh.read(_n))
_base=8+_n; _raw=np.memmap('/model/model.safetensors',dtype=np.uint8,mode='r')
def U(k):
    v=HDR[k]; s,e_=v['data_offsets']
    return np.frombuffer(_raw[_base+s:_base+e_],dtype=np.uint16).reshape(v['shape'])
def gap_profile(k,minhi=30):
    u=U(k); ee=((u>>7)&0xff).astype(int); mm=(u&0x7f).astype(int)
    hi=np.bincount(mm[ee>=123],minlength=128).astype(float)
    if hi.sum()<2000:
        hi=np.bincount(mm[ee>=122],minlength=128).astype(float)
    from scipy.ndimage import median_filter
    loc=median_filter(hi,size=9,mode='nearest')
    G=(hi<=0.04*loc)&(loc>=minhi)
    out={'nG':int(G.sum())}
    if G.sum()==0: return out
    idx=np.where(G)[0]
    for ex in range(116,125):
        s=mm[ee==ex]
        if len(s)<1500: out[ex]=np.nan; continue
        h_=np.bincount(s,minlength=128).astype(float)
        nb=[]
        for i in idx:
            for j in (i-2,i-1,i+1,i+2):
                if 0<=j<128 and not G[j]: nb.append(h_[j])
        out[ex]=round(float(h_[idx].mean()/max(np.mean(nb),1e-9)),3)
    return out
rows=[]
for k in NAMES:
    if 'norm' in k: continue
    r=gap_profile(k); s_=k.replace('model.','').replace('layers.','').replace('.weight','').replace('self_attn.','').replace('mlp.','')
    r['k']=s_; rows.append(r)
g=pd.DataFrame(rows).set_index('k'); g.to_pickle('/work/gapprof.pkl')
print(g.describe().round(3).loc[['count','mean','min','max']].to_string())
print(g[g.nG>0].sort_values(120).head(12).to_string())
print(g[g.nG>0].sort_values(122,ascending=False).head(12).to_string())
```