The resistant loops are not a new mechanism. In the edited model, the layer-20 onset-minus-novel difference has cosine 0.91 with the original clean direction, and its held-out separation is about the same (cv_d 0.91). So the edited model still enters loops along the direction C targets, and a C2 built here would just be C again. Higher dose already plateaued, so I'm closing the second-direction line.

One structural question remains. C removes the direction only from what layers 16–22 write, yet the onset signal is still present at layer 20. It may already be written before layer 16. I'll compare the mean onset-minus-novel difference projected on the clean direction at layers 16 and 20, in the original (fwd_8) and the edited model (fwd_36). These are only summary vectors. If most of the projection is already present at layer 16 in the edited model, extending C to earlier layers (12–15) is a justified variant. If it is mostly rebuilt between layers 16 and 20 despite the edit, earlier layers are not the source and I stop.

```python
def mdiff(fdir,ids_,onl,novl,L):
    H=np.load(f'{fdir}/hidden_{L}.npy',mmap_mode='r')
    ofirst=[int(ids_[i,q+1]) for i,q in onl]; nfirst=np.array([int(ids_[i,q+1]) for i,q in novl])
    top=[t for t,c in Counter(ofirst).most_common(8)]; O=[];N=[]
    for t in top:
        oo=[(i,q) for (i,q),f in zip(onl,ofirst) if f==t]; nn=[novl[k] for k in np.where(nfirst==t)[0]]
        if len(nn)<10: continue
        Om=np.stack([H[i,q].astype(np.float32) for i,q in oo]); Nm=np.stack([H[i,q].astype(np.float32) for i,q in nn])
        mu=Nm.mean(0); O.append(Om-mu); N.append(Nm-mu)
    O=np.concatenate(O); N=np.concatenate(N)
    return O.mean(0)-N.mean(0), float(np.linalg.norm(np.concatenate([O,N]).std(0)))
out=[]
for tag,fd,ids_,onl,novl in [('orig','/work/fwd_8',ids8,onset,novel),('edit','/work/fwd_36',ids36,on2,nov2)]:
    for L in [16,20]:
        d,sd=mdiff(fd,ids_,onl,novl,L)
        out.append(dict(model=tag,L=L,proj_clean=round(float(d@dclean),2),norm_diff=round(float(np.linalg.norm(d)),2),cos=round(float(d@dclean/np.linalg.norm(d)),2)))
pd.DataFrame(out).to_string()
```