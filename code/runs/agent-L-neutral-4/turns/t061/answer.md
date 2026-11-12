The stronger edit is not selective. At β = 3 the onset first token falls by 1.71 nats, but ordinary novel line starts fall by 0.95 and reference answer tokens by 0.25 nats per token, which would wreck reasoning. The selectivity (about 1.8:1) is the same as at β = 1. So the layer-20 onset direction is largely a generic "start a new line" feature with a weak repetition component, and removing or reversing it can't target loops. I'm closing L as constructed and restoring nothing yet; the current edit is L(3.0) and will be replaced.

One confound may explain the poor selectivity. Onset lines and novel lines start with different tokens: duplicated lines are mostly "The ..." restatements. I'll rebuild the direction contrasting onset newlines only with novel newlines whose next line begins with the same first token as the onset line. That removes the "which token comes next" part and leaves the "repeat versus new content" part. If this cleaner direction still separates well (cross-validated d above about 1) and differs from the old one (low cosine), it's worth one teacher-forced screen. If the separation vanishes, repetition isn't linearly marked at line starts beyond token identity, and I'll close the loop line.

```python
H=np.load('/work/fwd_8/hidden_20.npy',mmap_mode='r')
ofirst=[int(ids8[i,q+1]) for i,q in onset]; nfirst=np.array([int(ids8[i,q+1]) for i,q in novel])
from collections import Counter
cnt=Counter(ofirst); top=[t for t,c in cnt.most_common(8)]
print('onset first tokens covered by top8:',round(sum(cnt[t] for t in top)/len(onset),2))
O=[];N=[];oi=[];ni=[]
for t in top:
    oo=[(i,q) for (i,q),f in zip(onset,ofirst) if f==t]; nn=[novel[k] for k in np.where(nfirst==t)[0]]
    if len(nn)<10: continue
    Om=np.stack([H[i,q].astype(np.float32) for i,q in oo]); Nm=np.stack([H[i,q].astype(np.float32) for i,q in nn])
    O.append(Om); N.append(Nm-Nm.mean(0)+0); oi+= [i for i,_ in oo]; ni+=[i for i,_ in nn]
    O[-1]=Om-Nm.mean(0); N[-1]=Nm-Nm.mean(0)   # center both by same-first-token novel mean
O=np.concatenate(O); N=np.concatenate(N); oi=np.array(oi); ni=np.array(ni)
oa=np.isin(oi,list(half)); na=np.isin(ni,list(half))
dc=O[oa].mean(0)-N[na].mean(0); dc/=np.linalg.norm(dc)
po,pn=O[~oa]@dc,N[~na]@dc
dfull=O.mean(0)-N.mean(0)
print(dict(n_on=len(O),n_nov=len(N),cv_d=round(float((po.mean()-pn.mean())/np.sqrt((po.var()+pn.var())/2)),2),
           cos_with_d20=round(float(dfull@dirs[20]/np.linalg.norm(dfull)/np.linalg.norm(dirs[20])),2)))
```