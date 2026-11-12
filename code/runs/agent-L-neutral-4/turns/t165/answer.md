Pass 36 returned layer-16 and layer-20 residuals for all 512 C(−1.5) DEV answers run through the edited model. The per-text mean log-probs include prompt tokens, so they don't matter here. Now I'll apply the block-43 rule to find onset, novel and in-loop newline positions in these texts. Then I'll apply the block-50 matched-first-token recipe at layers 20 and 16. The quantities of interest are the number of onsets, the held-out separation (cv_d, split by text) and the cosine with the saved clean direction. A cosine above about 0.8 means the resistant loops start along the same direction C already removes, which would close the line. A clearly lower cosine with cv_d comparable to the original's would support a second edit.

```python
ids36=np.load('/work/fwd_36/input_ids.npy'); lp36=np.load('/work/fwd_36/token_logprobs.npy')
lens36=json.load(open('/work/fwd_36/meta.json'))['lengths']
plen36=[len(tok.encode(fmt.format(problem=prac[m['id']]['problem'])).ids)-1 for m in c15meta]
on2=[];nov2=[];inl2=[]
for i in range(len(c15meta)):
    a=ids36[i,:lens36[i]]; p0=plen36[i]
    nls=[q for q in range(p0,len(a)-1) if a[q] in NL]
    seen=set(); found=False
    for j,q in enumerate(nls):
        end=nls[j+1] if j+1<len(nls) else len(a)
        line=tuple(a[q+1:end])
        if len(line)>=4 and line in seen:
            (inl2 if found else on2).append((i,q)); found=True
        elif not found and len(line)>=4: nov2.append((i,q))
        seen.add(line)
print('onsets',len(on2),'novel',len(nov2),'inloop',len(inl2),'onset texts cut:',sum(c15meta[i]['cut'] for i,_ in on2))
otx=sorted(set(i for i,_ in on2)); rng=np.random.default_rng(0); half2=set(rng.choice(otx,len(otx)//2,replace=False))
dclean=np.load('/work/onset_dir_clean_L20.npy').astype(np.float32); dclean/=np.linalg.norm(dclean)
res=[]; dirs2={}
for L in [16,20]:
    H=np.load(f'/work/fwd_36/hidden_{L}.npy',mmap_mode='r')
    ofirst=[int(ids36[i,q+1]) for i,q in on2]; nfirst=np.array([int(ids36[i,q+1]) for i,q in nov2])
    cnt=Counter(ofirst); top=[t for t,c in cnt.most_common(8)]
    O=[];N=[];oi=[];ni=[]
    for t in top:
        oo=[(i,q) for (i,q),f in zip(on2,ofirst) if f==t]; nn=[nov2[k] for k in np.where(nfirst==t)[0]]
        if len(nn)<10: continue
        Om=np.stack([H[i,q].astype(np.float32) for i,q in oo]); Nm=np.stack([H[i,q].astype(np.float32) for i,q in nn])
        mu=Nm.mean(0); O.append(Om-mu); N.append(Nm-mu); oi+=[i for i,_ in oo]; ni+=[i for i,_ in nn]
    O=np.concatenate(O); N=np.concatenate(N); oi=np.array(oi); ni=np.array(ni)
    oa=np.isin(oi,list(half2)); na=np.isin(ni,list(half2))
    dc=O[oa].mean(0)-N[na].mean(0); dc/=np.linalg.norm(dc); po,pn=O[~oa]@dc,N[~na]@dc
    dfull=O.mean(0)-N.mean(0); dirs2[L]=dfull
    res.append(dict(L=L,n_on=len(O),n_nov=len(N),cover=round(sum(cnt[t] for t in top)/len(on2),2),
                    cv_d=round(float((po.mean()-pn.mean())/np.sqrt((po.var()+pn.var())/2)),2),
                    cos_clean=round(float(dfull@dclean/np.linalg.norm(dfull)),2)))
pd.DataFrame(res).to_string()
```