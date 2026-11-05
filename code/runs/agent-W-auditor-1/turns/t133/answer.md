Dimension 203 is a real excess, but it needs care. It's the top key gain in 4 layers, which only about 1 in 2,000 shuffles reproduces. Against that, I picked this test after noticing the pattern, and gain vectors correlate across layers (mean 0.115, max 0.43), so dimensions aren't fully interchangeable and the shuffled baseline is too generous.

Another detail undercuts it. Wherever the k gain is huge, the q gain at 203 is close to dead (L2 23.0 and 0.2, L6 20.8 and 0.3, L10 22.2 and 0.4). Only the product of q and k gains affects scores, so the size of k alone may just be a scale trade-off between the two gains.

There's a clean way to tell these apart. Layers shouldn't share dimension indices unless they share an origin, such as a common initialisation. So I'll compare `k_proj` rows with the same index across layers against rows with different indices. If same-index alignment shows up across all dimensions, 203 is a natural leftover of shared structure. If only 203 and a few others align, it's a specific channel and I'll look at what it reads. I'll also check whether the q-k gain trade-off is general, and which residual coordinates dimension 203 reads.

```python
import numpy as np
Ks=[]; Qm=[]
for l in range(18):
    K=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    Ks.append(K/np.linalg.norm(K,axis=1,keepdims=True).clip(1e-9))
# same-index vs different-index |cos| across layer pairs
same=np.zeros((18,18)); diff=np.zeros((18,18)); perdim=np.zeros(256); npairs=0
rng=np.random.RandomState(0); perm=rng.permutation(256)
for a in range(18):
    for b in range(a+1,18):
        c=np.abs((Ks[a]*Ks[b]).sum(1))
        same[a,b]=c.mean(); diff[a,b]=np.abs((Ks[a]*Ks[b][perm]).sum(1)).mean()
        perdim+=c; npairs+=1
perdim/=npairs
iu=np.triu_indices(18,1)
print('mean |cos| same index %.4f, different index %.4f, expected random %.4f'%(same[iu].mean(),diff[iu].mean(),np.sqrt(2/np.pi/640)))
print('dims with highest same-index |cos|',[(int(d),round(float(perdim[d]),3)) for d in np.argsort(-perdim)[:8]],'| dim 203',round(float(perdim[203]),3),'| median dim',round(float(np.median(perdim)),3))
# q-k gain trade-off inside a layer
rs=[]
for l in range(18):
    gq=np.clip(G['q'][l],1e-3,None); gk=np.clip(G['k'][l],1e-3,None)
    rs.append(round(float(np.corrcoef(np.log(gq),np.log(gk))[0,1]),2))
print('corr(log gq, log gk) inside layer',rs)
P=np.array(G['q'])*np.array(G['k'])
print('product gq*gk at 203',np.round(P[:,203],1).tolist())
print('layer median product',np.round(np.median(np.abs(P),axis=1),1).tolist())
for l in (2,6,7,10):
    x=tokens_x(l); gi=1+load_original(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    xn=x/np.sqrt((x**2).mean(1,keepdims=True))*gi
    K=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    ctr=K[203]*xn.mean(0); tot=(xn@K[203])
    order=np.argsort(-np.abs(ctr))[:4]
    print('L',l,'k203 mean %.2f sd %.2f'%(tot.mean(),tot.std()),'top coord contributions',[(int(j),round(float(ctr[j]),2)) for j in order])
```