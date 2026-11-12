Let loops have a clear shape. 93% contain an exact repeated line, the repeat begins about a quarter of the way through the answer, after a median of 5 equations, and the period varies from 1 to 8 lines. Most repeated lines start with "The", "We" or "$". The excerpts show the model re-deriving earlier statements after reaching a dead end, like the example that finds a difference of 0 and then starts over. It's a broad behaviour with no fixed trigger phrase, and it fits the earlier finding that the model usually doesn't have the answer yet at loop onset.

That makes a neat fix less likely, but one inexpensive question remains before I close it: is there a specific component behind the repetition, or does the state merely look different? Pass 2 has per-token MLP activations on 90 cut answers. So for each answer I can compare tokens inside the loop with tokens before onset, which holds the problem and style constant. I'll rank neurons at L13, L18 and L23 by a paired statistic across answers. A few neurons standing out sharply would be a real candidate for a loop circuit, which I could then remove at several strengths. A broad spread of small differences would mean no single component exists, and I'd close loop breaking without spending a pass.

```python
ma={l:np.load(f'/work/fwd_2/mlp_act_{l}.npy',mmap_mode='r') for l in (13,18,23)}
print({l:ma[l].shape for l in ma})
sets=[]
for i,x in enumerate(m2):
    if not x['cut']: continue
    ids_=np.asarray(ii2[i,:x['tlen']]).astype(int)
    st_=x['plen']; ln_=[]; cur=[]; start=st_
    for p in range(x['plen'],x['tlen']):
        if ids_[p] in (107,108):
            if cur: ln_.append((tuple(cur),start))
            cur=[]; start=p+1
        else: cur.append(int(ids_[p]))
    seen=set(); on=None
    for s,b in ln_:
        if len(s)>=5 and s in seen: on=b; break
        seen.add(s)
    if on is None or on-x['plen']<20 or x['tlen']-on<20: continue
    sets.append((i,on,x['tlen'],x['plen'],x['mode']))
print('usable loop answers',len(sets),Counter(s[4] for s in sets).most_common())
res={}
for l in ma:
    D_=[]
    for i,on,te,pl,md in sets:
        a_=np.asarray(ma[l][i,on:te]).astype(np.float32).mean(0); b_=np.asarray(ma[l][i,pl:on]).astype(np.float32).mean(0)
        D_.append(a_-b_)
    D_=np.stack(D_); me=D_.mean(0); sd=D_.std(0,ddof=1)+1e-6; t=me/sd*np.sqrt(len(D_))
    res[l]=(me,sd,t)
    top=np.argsort(-np.abs(t))[:6]
    print('L',l,'top |t|',[(int(j),round(float(t[j]),1),round(float(me[j]),3)) for j in top],'| n |t|>6:',int((np.abs(t)>6).sum()),'of',len(t),'q99 |t| %.2f'%float(np.quantile(np.abs(t),.99)))
```