Block 25 confirms the depth profile excluded coordinates 1038 and 1125 and rescaled each vector by its own RMS. So my "shift" measure couldn't see a volume difference, and I've never tested the volume-knob idea between contexts. It also shows block 15 flagged a list of outlier MLP neurons, summarised only as "look natural", and I no longer have those numbers. That summary was a quick judgement, so I'll look at them again.

This run uses only stored data and two checks:
1. The values of coordinates 1038 and 1125 by layer in eval vs raw, and each one's share of residual energy, on matched tokens.
2. The flagged-neuron table again.

If eval has consistently larger 1038 energy, and therefore weaker effective signal, at the layers where digits get computed, that supports a gain mechanism I can test directly. If the two contexts match within a few percent, I can close the idea. An outlier neuron worth a look would be one with large weights and a clear link to failure.

```python
import numpy as np, pandas as pd
print('flagged (layer,stat,count,first idx):', flagged[:40])
Hx = H.copy()
print(Hx[['l','j','zg','zu','zd','g_top','g_topsh','u_top','u_topsh','d_top','d_topsh','mA','sA','mP','dcorr','dcut','share']].round(2).head(30).to_string())

# volume knob: matched tokens eval vs raw
rows_v=[]
ri = res_idx[:1500] if len(res_idx)>1500 else res_idx
oi = oth_s
for l in list(range(0,27,2))+[11,12,13]:
    H_=getH(l)
    for nm,idxs_ in [('res',ri),('oth',oi)]:
        r=PT.loc[idxs_]
        he_=np.array([H_[p*2,t-1] for p,t in zip(r.pi,r.te)],dtype=np.float32)
        hr_=np.array([H_[p*2+1,t-1] for p,t in zip(r.pi,r.tr)],dtype=np.float32)
        def st(h):
            tot=(h**2).sum(1); rest=tot-(h[:,1038]**2)-(h[:,1125]**2)
            return h[:,1038], h[:,1125], np.sqrt(rest/1150), (h[:,1038]**2)/tot
        a1,b1,c1_,s1=st(he_); a2,b2,c2_,s2=st(hr_)
        rows_v.append(dict(l=l,tok=nm,c1038_e=a1.mean(),c1038_r=a2.mean(),sd1038_e=a1.std(),c1125_e=b1.mean(),c1125_r=b2.mean(),restRMS_e=c1_.mean(),restRMS_r=c2_.mean(),ratio_sig=(c1_/np.abs(a1)).mean()/(c2_/np.abs(a2)).mean(),share_e=s1.mean(),share_r=s2.mean()))
V_=pd.DataFrame(rows_v).sort_values(['tok','l'])
print(V_.round(3).to_string(index=False))
```