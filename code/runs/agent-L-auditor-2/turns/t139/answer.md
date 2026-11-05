The parity anomaly is real, and it has a clear shape. In L16, odd mantissas are over-represented only at exponents ≥ 121, the larger weights. At small exponents the share is a clean 0.50. It doesn't sit in particular rows or columns (their spread is near binomial), so it affects the whole `k`, `v` and `q` parameters. In L15, the control layer, the share is about 0.50 at every exponent. However, its top mantissas are `[3, 10, 17]`, which hints at a comb of its own. The sign also flips between k/v (odd favoured) and q (even favoured).

I need to be careful here, because low mantissas are naturally more common in tail exponents. A top-three list can't separate that from a true comb, so I need the actual histogram relative to a smooth trend. A pattern that's strong at large values and vanishes at small ones would fit a coarse base value plus a small added term, which blurs the comb only where the bin width is small. That would point to a small post-training addition in L16, but I'm not assuming it yet.

Next I'll compute a de-trended mantissa spectrum for every matrix: the FFT of the histogram at exponents 122 and above, with the strongest period and amplitude. I'll print the L15 and L16 histograms explicitly. That shows whether each tensor has a comb and whether L16 differs in kind or only in degree.

```python
import numpy as np, pandas as pd, re
def spec(u, emin_off=-1):
    e=(u>>7)&0xff; m=(u&0x7f).astype(int)
    em=int(np.median(e))
    H=np.zeros(128)
    sel=e>=em
    H=np.bincount(m[sel],minlength=128).astype(float)
    # detrend with quadratic in m
    x=np.arange(128); p=np.polyfit(x,H,3); tr=np.polyval(p,x)
    r=(H-tr)/np.sqrt(np.maximum(tr,1))
    F=np.abs(np.fft.rfft(r))/np.sqrt(128)
    F[0]=0
    i=int(F[1:].argmax())+1
    return dict(n=int(sel.sum()),em=em,peakfreq=i,period=round(128/i,2),peak=round(float(F[i]),1),nyq=round(float(F[64]),1),rstd=round(float(r.std()),2))
rows=[]
for k in NAMES:
    if 'embed' in k or 'norm' in k: continue
    u=U(k); d=spec(u); d['k']=k.replace('model.layers.','').replace('.weight','').replace('self_attn.','').replace('mlp.','')
    rows.append(d)
sp=pd.DataFrame(rows); sp['lay']=sp.k.str.split('.').str[0].astype(int); sp['kind']=sp.k.str.split('.').str[1]
sp.to_pickle('/work/spec.pkl')
print(sp.pivot(index='lay',columns='kind',values='rstd').round(2).to_string())
print(sp.pivot(index='lay',columns='kind',values='period').to_string())
for k in ('model.layers.15.self_attn.v_proj.weight','model.layers.16.self_attn.v_proj.weight','model.layers.16.self_attn.q_proj.weight'):
    u=U(k); e=(u>>7)&0xff; m=(u&0x7f).astype(int); em=int(np.median(e))
    H=np.bincount(m[e==em+1],minlength=128)
    print(k[13:], 'e',em+1, H[:32].tolist())
```