The token-level contrast is clearer than the answer-mean one. Copied tokens are near-deterministic (mean logprob -0.013 vs -0.117 for novel tokens), and a few neurons, concentrated in layers 9-15, separate copied from novel tokens with effect sizes of 1.0-1.3. The 99th percentile of the effect size is only 0.5-0.7, so these are clear outliers. The top ones (L13 n486, n1926, n4180; L9 n5659; L15 n386, n5730) are the same units that topped the answer-mean comparison, which is a consistency check. One confound remains: copied tokens sit later in the text than novel ones. So I'll check the position dependence of these neurons on the 8 finished controls, which have no loops, early vs late tokens. I'll also count how many neurons pass an effect-size threshold, to size an ablation set. If the top neurons are flat with position in the controls, they track repetition itself and are worth an ablation test.

```python
rep_cands=[]
for l in [5,7,9,11,13,15,17,19,21,23,25]:
    X=np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r')
    s1=np.zeros(6912);s2=np.zeros(6912);n1=0;t1=np.zeros(6912);t2=np.zeros(6912);n2=0
    for i in range(24):
        xi=X[i].astype(np.float32); a=xi[cm[i]]; b=xi[nm[i]]
        s1+=a.sum(0); s2+=(a**2).sum(0); n1+=len(a); t1+=b.sum(0); t2+=(b**2).sum(0); n2+=len(b)
    m1=s1/n1; m2=t1/n2; v=(s2+t2)/(n1+n2)-((s1+t1)/(n1+n2))**2; d=(m1-m2)/np.sqrt(v+1e-8)
    # position control in finished texts: early (first half of answer) vs late
    e=[];la=[]
    for i in range(24,32):
        xi=X[i].astype(np.float32); s=plen[i]; mid=(s+L[i])//2
        e.append(xi[s:mid]); la.append(xi[mid:L[i]])
    e=np.concatenate(e); la=np.concatenate(la); dpos=(la.mean(0)-e.mean(0))/np.sqrt(v+1e-8)
    for j in np.where(np.abs(d)>0.8)[0]:
        rep_cands.append(dict(l=l,n=int(j),d=round(float(d[j]),2),dpos=round(float(dpos[j]),2),m_copy=float(m1[j]),m_novel=float(m2[j])))
rc=pd.DataFrame(rep_cands)
json.dump(rep_cands, open('/work/rep_cands.json','w'))
(len(rc), rc.groupby('l').size().to_dict(), rc.sort_values('d',key=abs,ascending=False).head(15)[['l','n','d','dpos']].to_string())
```