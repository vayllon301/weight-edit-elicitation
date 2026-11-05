The group is cleanly defined: 159 dimensions, 152 of them low in all seven early layers, and all 120 sink dimensions are included. The embedding doesn't favour these dimensions, with variance of 0.000715 against 0.000726, so the channel isn't simply a copy of the input and must be filled by what layers 0-4 write.

Now I'll test the key claim directly on the per-token states from pass 3, which cover all layers on 200 texts. For the B dimensions and the rest, I'll measure three things:

1. **Size:** the token-to-token variation at each layer.
2. **Persistence:** the correlation of each dimension across tokens between layer pairs such as 5 and 16. A true bypass should stay high in B across the middle while the other dimensions decorrelate.
3. **Shallowness:** the share of variance explained by the current token's identity alone. A high share at layer 16 would mean late layers see something close to a word-level feature, and a low share would mean B holds real context information.

A sharp contrast between B and the rest would confirm the circuit and give me a specific reason to test lesions. If B turns over like everything else, the bypass reading is wrong and I'd only test the sink.

```python
import numpy as np, json
ids=np.load('/work/fwd_3/input_ids.npy',mmap_mode='r'); lens=np.array(m['lengths'])
mask=np.zeros(ids.shape,bool)
for i,n in enumerate(lens): mask[i,:min(n,ids.shape[1])]=True
ti,tj=np.where(mask); rng=np.random.default_rng(3); sel=rng.permutation(len(ti))[:12000]; ti2,tj2=ti[sel],tj[sel]
order=np.lexsort((tj2,ti2)); ti2,tj2=ti2[order],tj2[order]
Lset=[0,1,3,5,8,12,16,17,18,20,22,24,26]
H={}
for l in Lset:
    a=hs(l); H[l]=np.asarray(a[ti2,tj2]).astype(np.float32)
tok=np.asarray(ids[ti2,tj2])
print('tokens',len(tok),'distinct',len(np.unique(tok)))
print('layer: sd per dim (median) B, R, mean|dim1038|')
for l in Lset:
    s=H[l].std(0); print(l,round(float(np.median(s[B])),2),round(float(np.median(s[R])),2),round(float(np.abs(H[l][:,1038]).mean()),0))
def pc(a,b,dims):
    x=H[a][:,dims]; y=H[b][:,dims]; x=(x-x.mean(0))/(x.std(0)+1e-6); y=(y-y.mean(0))/(y.std(0)+1e-6)
    return float(np.median((x*y).mean(0)))
print('\npersistence corr median per dim: pair B R')
for a,b in [(1,3),(3,5),(5,8),(8,12),(12,16),(5,16),(3,16),(16,17),(16,18),(16,20),(20,24),(5,24),(24,26)]:
    print((a,b),round(pc(a,b,B),3),round(pc(a,b,R),3))
u,inv,c=np.unique(tok,return_inverse=True,return_counts=True); keep=c[inv]>=4
print('\neta2 token identity (tokens with count>=4, n=%d): layer B R'%keep.sum())
for l in [1,3,5,8,12,16,20,24,26]:
    X=H[l][keep]; lab=inv[keep]; nl=lab.max()+1
    g=X.mean(0); ss_t=((X-g)**2).sum(0)
    sm=np.zeros((nl,1152),np.float64); np.add.at(sm,lab,X-g); ct=np.bincount(lab,minlength=nl)[:,None]; ss_b=(sm**2/np.maximum(ct,1)).sum(0)
    e=ss_b/(ss_t+1e-6); print(l,round(float(np.median(e[B])),3),round(float(np.median(e[R])),3))
```