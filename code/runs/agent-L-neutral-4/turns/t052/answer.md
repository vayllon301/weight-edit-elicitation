Pass 8 returned hidden states for all 450 texts. Now the labelling, from the returned input_ids:
- Newline tokens in the answer split it into lines.
- A line of at least 4 tokens that exactly repeats an earlier answer line is a duplicate. The newline just before the first duplicate is the onset position.
- Newlines before novel lines are controls: those before the onset in looping texts, and all of them in finished texts.

At each layer I'll compute the difference of means (onset minus same-text novel). To check that it is a real signal and not noise from about 200 onsets, I'll estimate the direction on half the texts and measure the effect size d on the other half. I'll also report the onset token's logprob to see how soft the decision is. A cross-validated d well above about 1 at some layer would identify a usable onset direction; a d near 0 would close this line.

```python
ids8=np.load('/work/fwd_8/input_ids.npy'); lp8=np.load('/work/fwd_8/token_logprobs.npy')
m8=json.load(open('/work/fwd_8/meta.json')); lens8=m8['lengths']
NL=set(i for i in range(tok.get_vocab_size()) if (s:=tok.decode([i])) and set(s)<= {'\n'})
plen=[len(tok.encode(fmt.format(problem=prac[mm['id']]['problem'])).ids)-1 for mm in meta]
onset=[];novel=[];inloop=[]   # (text, pos)
for i in range(len(meta)):
    a=ids8[i,:lens8[i]]; p0=plen[i]
    nls=[q for q in range(p0,len(a)-1) if a[q] in NL]
    seen=set(); found=False
    for j,q in enumerate(nls):
        end=nls[j+1] if j+1<len(nls) else len(a)
        line=tuple(a[q+1:end])
        if len(line)>=4 and line in seen:
            (inloop if found else onset).append((i,q)); found=True
        elif not found and len(line)>=4: novel.append((i,q))
        seen.add(line)
print(len(onset),len(novel),len(inloop))
olp=np.array([lp8[i,q] for i,q in onset]); nlp=np.array([lp8[i,q] for i,q in novel]); ilp=np.array([lp8[i,q] for i,q in inloop])
print('onset first-token lp median',np.median(olp).round(2),'P(lp>-0.1)',(olp>-0.1).mean().round(2),'novel median',np.median(nlp).round(2),'inloop median',np.median(ilp).round(2))
otexts=sorted(set(i for i,_ in onset)); rng=np.random.default_rng(0); half=set(rng.choice(otexts,len(otexts)//2,replace=False))
res=[]; dirs={}
for L in [8,12,16,20,23]:
    H=np.load(f'/work/fwd_8/hidden_{L}.npy',mmap_mode='r')
    O=np.stack([H[i,q].astype(np.float32) for i,q in onset]); N=np.stack([H[i,q].astype(np.float32) for i,q in novel])
    oi=np.array([i for i,_ in onset]); ni=np.array([i for i,_ in novel])
    oa=np.isin(oi,list(half)); na=np.isin(ni,list(half))
    d=O[oa].mean(0)-N[na].mean(0); d/=np.linalg.norm(d)
    po,pn=O[~oa]@d,N[~na]@d
    res.append(dict(L=L,cv_d=round(float((po.mean()-pn.mean())/np.sqrt((po.var()+pn.var())/2)),2),
                    diffnorm_rel=round(float(np.linalg.norm(O.mean(0)-N.mean(0))/np.linalg.norm(N.mean(0))),3)))
    dirs[L]=(O.mean(0)-N.mean(0))
pd.DataFrame(res).to_string()
```