L was the unmatched layer-20 onset direction, and it was closed. No direction for the in-loop state (newlines before repeats after the onset) has been built or tested, so this line is open.

I'll take the block-50 recipe: in-loop newline residuals minus same-first-token novel means, held-out split by text. I'll compute it in the edited model (fwd_36, the loops that resist C) and in the original (fwd_8). The question is whether it differs from the clean onset direction. A low cosine with dclean plus strong held-out separation would make it a distinct candidate. I'd then screen both signs with a teacher-forced pass: the target is a lower probability of continuing a loop with no damage to reference solutions. A high cosine means this is the same direction and I close the line.

```python
def mdir(fdir,ids_,posl,novl,L,halfset):
    H=np.load(f'{fdir}/hidden_{L}.npy',mmap_mode='r')
    pf=[int(ids_[i,q+1]) for i,q in posl]; nfirst=np.array([int(ids_[i,q+1]) for i,q in novl])
    cnt=Counter(pf); top=[t for t,c in cnt.most_common(8)]; O=[];N=[];oi=[];ni=[]
    for t in top:
        oo=[(i,q) for (i,q),f in zip(posl,pf) if f==t]; nn=[novl[k] for k in np.where(nfirst==t)[0]]
        if len(nn)<10: continue
        Om=np.stack([H[i,q].astype(np.float32) for i,q in oo]); Nm=np.stack([H[i,q].astype(np.float32) for i,q in nn])
        mu=Nm.mean(0); O.append(Om-mu); N.append(Nm-mu); oi+=[i for i,_ in oo]; ni+=[i for i,_ in nn]
    O=np.concatenate(O); N=np.concatenate(N); oi=np.array(oi); ni=np.array(ni)
    oa=np.isin(oi,list(halfset)); na=np.isin(ni,list(halfset))
    dc=O[oa].mean(0)-N[na].mean(0); dc/=np.linalg.norm(dc); po,pn=O[~oa]@dc,N[~na]@dc
    d=O.mean(0)-N.mean(0)
    return d, dict(n=len(O),cover=round(sum(cnt[t] for t in top)/len(posl),2),cv_d=round(float((po.mean()-pn.mean())/np.sqrt((po.var()+pn.var())/2)),2),
                   cos_clean=round(float(d@dclean/np.linalg.norm(d)),2),proj=round(float(d@dclean),1),norm=round(float(np.linalg.norm(d)),1))
r=[]; indirs={}
for tag,fd,ids_,pl,nv in [('orig','/work/fwd_8',ids8,inloop,novel),('edit','/work/fwd_36',ids36,inl2,nov2)]:
    tx=sorted(set(i for i,_ in pl)); hs=set(np.random.default_rng(0).choice(tx,len(tx)//2,replace=False))
    d,s=mdir(fd,ids_,pl,nv,20,hs); indirs[tag]=d; s['model']=tag; r.append(s)
r.append(dict(model='cos orig-vs-edit inloop',cv_d=round(float(indirs['orig']@indirs['edit']/np.linalg.norm(indirs['orig'])/np.linalg.norm(indirs['edit'])),2)))
pd.DataFrame(r).to_string()
```