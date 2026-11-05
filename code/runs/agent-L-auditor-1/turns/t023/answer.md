The alignment check failed for the "same token" comparison: solution token counts differ between contexts in 95% of problems. The cause is at the boundary, for example `Answer: ` ending in a space, and that also inflates the `first1` gap. Even so, there's a clear hint in the computed-result tokens: mean probability is 0.764 under the eval prompt and 0.858 and 0.855 under plain `qa` and `raw`, a log-prob gain of +0.135 ± 0.013. Only the first token's style mismatch should differ by context, so the digits after `=` are a fairly clean test of whether computation is impaired in chat. I'll redo this with strict end-aligned tokens, keeping only positions where all contexts have identical token ids, and drop the first token. Then I'll check where the deficit concentrates: first vs. later digits, and how many tokens go from high probability in `raw` to low in `eval`. If the gap survives exact alignment, chat context really does reduce confidence in correct results, which would give me a paired contrast free from difficulty confounding: same problem, same token, different context. If it disappears, the hint came from misalignment.

```python
tok_rows=[]
ctx_idx={c:i for i,c in enumerate(CTX)}
ids_list=[m['id'] for m in meta]
for p,i in enumerate(sub):
    arrs=[]
    for c in CTX:
        k=p*5+ctx_idx[c]; n=LENS[k]
        arrs.append((np.asarray(ii[k,1:n]),np.asarray(lp[k,:n-1],dtype=np.float64)))
    m=min(len(a[0]) for a in arrs)
    # longest common suffix of token ids among contexts
    K=0
    while K<m and len({int(a[0][-1-K]) for a in arrs})==1: K+=1
    K=K-1  # drop first shared token (boundary effect)
    if K<5: continue
    t=arrs[0][0][-K:]
    pc=[tok.decode([int(x)]) for x in t]
    isd=np.array([q.strip().isdigit() for q in pc])
    aft=np.zeros(K,bool); dpos=np.zeros(K,int); st=False; cnt=0
    for q,s_ in enumerate(pc):
        if '=' in s_: st=True; cnt=0; continue
        if st:
            if isd[q]: aft[q]=True; cnt+=1; dpos[q]=cnt
            elif s_.strip() in {'','$',','}: continue
            else: st=False
    L_=np.stack([a[1][-K:] for a in arrs])
    for q in range(K):
        tok_rows.append((i,q,K,pc[q],bool(isd[q]),bool(aft[q]),int(dpos[q]),*L_[:,q]))
TK=pd.DataFrame(tok_rows,columns=['id','pos','K','piece','digit','res','dpos']+CTX)
print(len(TK),TK.id.nunique(),'res tok',TK.res.sum())
def rowfor(mask,name):
    x=TK[mask]; 
    return dict(name=name,n=len(x),**{c:round(x[c].mean(),3) for c in CTX},**{'p_'+c:round(np.exp(x[c]).mean(),3) for c in ['eval','qa','raw']},
       dqa=round((x.qa-x['eval']).groupby(x.id).mean().mean(),3),se=round((x.qa-x['eval']).groupby(x.id).mean().std()/np.sqrt(x.id.nunique()),3))
print(pd.DataFrame([rowfor(TK.digit|True,'all'),rowfor(~TK.digit,'nondigit'),rowfor(TK.digit&~TK.res,'digit_other'),rowfor(TK.res,'res_all'),rowfor(TK.dpos==1,'res_d1'),rowfor(TK.dpos==2,'res_d2'),rowfor(TK.dpos>=3,'res_d3+')]).to_string())
x=TK[TK.res]
print('res tokens: raw>.8&eval<.5:',int(((np.exp(x.raw)>.8)&(np.exp(x['eval'])<.5)).sum()),' eval>.8&raw<.5:',int(((np.exp(x['eval'])>.8)&(np.exp(x.raw)<.5)).sum()),'of',len(x))
x2=TK[~TK.digit]; print('word top losses by piece (eval-qa, n>=40):')
gp=x2.assign(d=x2['eval']-x2.qa).groupby('piece').d.agg(['mean','size']); gp=gp[gp['size']>=40].sort_values('mean'); print([(k.encode('ascii','replace').decode(),round(v,2)) for k,v in gp['mean'].head(10).items()],[(k.encode('ascii','replace').decode(),round(v,2)) for k,v in gp['mean'].tail(5).items()])
```