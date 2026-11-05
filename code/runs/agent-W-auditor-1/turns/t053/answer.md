The pass returned hidden states for all 680 references. It also reported 7,765 values overflowing float16 to infinity, so some residual coordinates exceed 65,504. Those are presumably the reservoir coordinates I found earlier, and I'll mask them instead of letting them corrupt the projections.

Before trusting the ledger, I'll validate it. The final-layer logit gap computed from `hidden_18`, the final norm and the embeddings must match the logprob gap the tool reported at the same positions. If it does, I'll attribute that gap to the embedding and each layer's update, at digit failures and at comparable successes. The test: one layer turning against the correct digit at failures but not at successes would be a specific suspect. Evidence for and against spread smoothly across layers would close this approach and point to a general capacity limit.

```python
import numpy as np, json
d7='/work/fwd_7/'
sel=np.array(json.load(open('/work/tf7_map.json'))); inv={int(v):k for k,v in enumerate(sel)}
R=np.load('/work/tf7_digit_recs.npy')
keep=np.isin(R[:,0].astype(int),sel); R7=R[keep].copy(); R7[:,0]=[inv[int(v)] for v in R7[:,0]]
E=load('model.embed_tokens.weight').numpy() if hasattr(load('model.embed_tokens.weight'),'numpy') else np.array(load('model.embed_tokens.weight'))
gw=load('model.norm.weight'); gw=gw.numpy() if hasattr(gw,'numpy') else np.array(gw)
print('final norm weight mean %.3f min %.3f max %.3f'%(gw.mean(),gw.min(),gw.max()),'E',E.shape)
H={}
ii=R7[:,0].astype(int); tt=R7[:,1].astype(int)-1
fail=(R7[:,4]==0)&(R7[:,9]==1); succ=R7[:,4]==1
rng=np.random.RandomState(1); fi=np.where(fail)[0]; si=rng.choice(np.where(succ)[0],4000,replace=False)
idx=np.concatenate([fi,si]); order=np.lexsort((tt[idx],ii[idx])); idx=idx[order]
Hs=np.zeros((19,len(idx),640),np.float32)
for l in range(19):
    a=np.load(d7+f'hidden_{l}.npy',mmap_mode='r')
    # chunk by text to keep reads sane
    for j in np.unique(ii[idx]):
        m=np.where(ii[idx]==j)[0]; Hs[l,m]=a[j,tt[idx][m]].astype(np.float32)
print('inf share by layer',[round(float(np.isinf(Hs[l]).mean()),5) for l in range(19)])
infc=np.isinf(Hs).any(axis=(0,1)); print('coords with inf',np.where(infc)[0][:20],'abs max finite by layer',[int(np.nanmax(np.where(np.isinf(Hs[l]),np.nan,np.abs(Hs[l])))) for l in (0,6,12,18)])
np.save('/work/tf7_Hs.npy',Hs); np.save('/work/tf7_idx.npy',idx)
print('final hidden rms med',np.median(np.sqrt(np.nanmean(np.where(np.isinf(Hs[18]),np.nan,Hs[18])**2,axis=1))))
```