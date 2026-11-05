The ledger is informative. Layers before index 12 contribute essentially nothing to digit margins. Evidence is written almost entirely at stream indices 12-13 (`model.layers[11]`, the full-attention layer, and `[12]`) and then 16-18. For number copying or choice (`later_digit`, `other_first`), success and failure diverge sharply at 12-13: +3 to +4 at successes, -1 to -2 at failures. For arithmetic results (`first_after_eq`), the deficit is spread out: index 13 about 2.2, 16 about 2.5, 17 about 1.4, 18 about 1.7. So no single layer turns against the right digit on arithmetic, which counts against one planted arithmetic switch.

Two patterns need explaining. Indices 14 and 17 run opposite to the winner in `other_first`: -1.34 and -0.97 at successes, +1.02 and +0.43 at failures. That looks like shrinkage of whichever digit leads, and it could be either natural or a real brake. The abrupt onset at index 12 could also come from simple scale growth along the stack, or from something structural.

So I'll check three things for free. First, update magnitudes and post-norm gains by layer, excluding coordinate 163, to see whether the onset is smooth or a jump. Second, whether indices 14 and 17 correlate negatively with the sum of indices 12 and 13 position by position, which would confirm shrinkage. Third, the digit confusion matrix at arithmetic failures, in case a specific digit bias is at work. These are descriptive statistics on the unedited model only.

```python
import numpy as np
np.set_printoptions(linewidth=200,precision=2,suppress=True)
samp=np.random.RandomState(2).choice(len(idx),1500,replace=False)
rows=[]
pv=np.array(Hs[0][samp]); 
mask=np.ones(640,bool); mask[163]=False
for l in range(1,19):
    cu=np.array(Hs[l][samp]); dh=cu-pv
    dh2=np.where(np.isfinite(dh),dh,np.nan)[:,mask]
    pa=load(f'model.layers.{l-1}.post_attention_layernorm.weight'); pf=load(f'model.layers.{l-1}.post_feedforward_layernorm.weight')
    pa=np.array(pa)[mask]; pf=np.array(pf)[mask]
    ch=np.nanmedian(np.where(np.isfinite(cu[:,163]),cu[:,163],np.nan))
    rows.append((l,round(float(np.sqrt(np.nanmean(dh2**2))),1),round(float(np.sqrt(np.nanmean(np.where(np.isfinite(cu),cu,np.nan)[:,mask]**2))),1),round(float(np.median(np.abs(1+pa))),2),round(float(np.median(np.abs(1+pf))),2),round(float(np.abs(1+pa).max()),0),round(float(np.abs(1+pf).max()),0),int(ch)))
    pv=cu
print('idx,updRMS,stateRMS(no163),med|gain_attn|,med|gain_ff|,max_attn,max_ff,c163')
for r_ in rows: print(r_)
# anti-winner test: position-wise correlation
v=valid
for k in [0,1,2]:
    m=v&(cls==k)
    a=D[m][:,12:14].sum(1)
    print(nm[k],'corr(D14,D12+13) %.2f corr(D17,D12+13) %.2f corr(D15,..) %.2f corr(D16,..) %.2f corr(D18,..) %.2f'%tuple(np.corrcoef(D[m][:,q_],a)[0,1] for q_ in (14,17,15,16,18)))
# digit confusion at arithmetic fails
inv_d={k:int(v_) for k,v_ in digit_ids.items()}
mk=valid&isfail&(cls==0)
Cm=np.zeros((10,10),int)
for a_,b_ in zip(c[mk],w[mk]):
    if int(b_) in inv_d: Cm[inv_d[int(a_)],inv_d[int(b_)]]+=1
print('rows=right digit, cols=model top digit');print(Cm)
allc=np.bincount([inv_d[int(a_)] for a_ in c[valid&(cls==0)]],minlength=10)
print('right digit freq',allc,'pred freq at fails',Cm.sum(0),'fail rate by right digit',np.round(Cm.sum(1)/np.maximum(allc,1),2))
```