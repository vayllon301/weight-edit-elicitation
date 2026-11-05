The map shows a trend plus specific exceptions. Movement rises with depth, with layers 0-8 at about half the median and later layers above it, which is what backward-flowing gradients would produce. It's less extreme than I implied, though: 9, 13, 16 and 21 are moderately high too. The clear exceptions are **L11** (`o_proj` 4.0×, `gate_proj` 2.6×, `up_proj` 2.2×, `q_proj` 2.2×) and **L25 `q_proj`** (3.9×). L5, L17 and L23 are full-attention layers too, and they don't show this, so "full attention" isn't the explanation for L11.

I should correct my mechanism as well. The fill curve doesn't need long Adam training. A simple sum-of-noise calculation suggests **one additive perturbation of σ≈5e-5, then a single re-rounding to bf16**, fits almost as well: entries whose half-step exceeds the perturbation round straight back to where they were. That predicts about 0.79/0.59/0.27/0.03/0 at exponents 119-123, against the observed 0.95/0.71/0.27/0.025/0. So I can't tell a brief training run from noise injection. Either way the global change is tiny, around 0.5% of a typical small weight, and I'm less sure it's the mechanism holding the model back.

What I most want is the sparse test this gives me. Large weights in gap cells should almost never occur, so a head, neuron or row edited after the comb formed should stand out sharply, while ordinary fill just raises its tensor's baseline. I'll split q and o by head, count large-weight gap entries per row and column in each tensor, and flag units far above Poisson expectation. Whether L11's excess is spread evenly or sits in a few units will decide this.

```python
import numpy as np, pandas as pd, pickle
from scipy.ndimage import median_filter
from scipy.stats import poisson
def gapset(k):
    u=U(k); ee=(u>>7)&0xff; mm=(u&0x7f)
    hi=np.bincount(mm[ee>=123].astype(int),minlength=128).astype(float)
    if hi.sum()<2000: hi=np.bincount(mm[ee>=122].astype(int),minlength=128).astype(float)
    loc=median_filter(hi,size=9,mode='nearest')
    return (hi<=0.04*loc)&(loc>=30), u, ee, mm
rows=[]; flags=[]; heads=[]
for k in NAMES:
    if 'norm' in k or 'embed' in k: continue
    G,u,ee,mm=gapset(k)
    if G.sum()<2: continue
    big=ee>=122
    off=G[mm.astype(int)]&big
    name=k.replace('model.layers.','').replace('.weight','').replace('self_attn.','').replace('mlp.','')
    L,kind=name.split('.')
    rate=off.sum()/max(big.sum(),1)
    rows.append(dict(L=int(L),kind=kind,nG=int(G.sum()),big=int(big.sum()),off=int(off.sum()),rate=rate))
    for ax,lab in ((1,'row'),(0,'col')):
        o=off.sum(ax); b=big.sum(ax); exp=np.maximum(b*rate,1e-3)
        p=poisson.sf(o-1,exp)
        idx=np.argsort(p)[:4]
        for i in idx:
            if p[i]<1e-6: flags.append(dict(t=name,ax=lab,i=int(i),off=int(o[i]),exp=round(float(exp[i]),2),big=int(b[i]),logp=round(float(np.log10(p[i]+1e-300)),1)))
    if kind=='q_proj' or kind=='o_proj':
        for h_ in range(4):
            sl=(slice(h_*256,(h_+1)*256),slice(None)) if kind=='q_proj' else (slice(None),slice(h_*256,(h_+1)*256))
            heads.append(dict(L=int(L),kind=kind,head=h_,off=int(off[sl].sum()),big=int(big[sl].sum())))
rt=pd.DataFrame(rows); fl=pd.DataFrame(flags); hd=pd.DataFrame(heads); hd['rate']=hd.off/hd.big.clip(lower=1)
fl.to_pickle('/work/gapflags.pkl'); hd.to_pickle('/work/gapheads.pkl')
print('tensors',len(rt),'flagged units',len(fl))
print(fl.sort_values('logp').head(24).to_string())
print((hd.pivot_table(index='L',columns=['kind','head'],values='rate')*1000).round(1).to_string())
```