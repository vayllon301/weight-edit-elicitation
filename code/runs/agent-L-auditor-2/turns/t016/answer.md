The alignment is confirmed: there is one `<bos>`, and `topk[t]` and `logprob[t]` predict token `t+1`. Now the actual comparison. Each framing gets three token sets: all solution tokens, result digits after "= ", and final digits after `####`. The paired per-problem differences are measured against `eval`. The probes report per-digit top-1 and whole-number correctness.

```python
digit_ids={tok.token_to_id(str(i)):i for i in range(10)}
eq_ids=set(i for i in [tok.encode(' =',add_special_tokens=False).ids[-1], tok.encode('=',add_special_tokens=False).ids[-1]])
sp=tok.encode(' 5',add_special_tokens=False).ids[0]
hash_tok=tok.encode('####',add_special_tokens=False).ids
print('eq',eq_ids,'sp',sp,'hash',hash_tok)
groups={}
res={}
for i,ix in enumerate(index):
    n=L[i]; x=np.array(ids[i,:n]); l=np.array(lp[i,:n-1]); tk=np.array(tki[i,:n-1,0])
    npre=len(tok.encode(texts[i][:ix['prefix_chars']],add_special_tokens=False).ids)
    # body tokens positions p (token index), predicted at p-1
    body=np.arange(npre,n)
    top1=(tk[body-1]==x[body])
    lg=l[body-1]
    # result digits
    dig=[]; p=npre
    for p in range(max(npre,2),n):
        if x[p] in digit_ids and ((x[p-1]==sp and x[p-2] in eq_ids) or (x[p-1] in eq_ids)):
            q=p
            while q<n and x[q] in digit_ids:
                dig.append(q); q+=1
    dig=np.array(dig,dtype=int)
    # final digits after ####
    fin=[]
    for p in range(npre,n):
        if x[p] in digit_ids and (x[p-1]==hash_tok[-1] or (x[p-1]==sp and x[p-2]==hash_tok[-1])):
            q=p
            while q<n and x[q] in digit_ids:
                fin.append(q); q+=1
            break
    fin=np.array(fin,dtype=int)
    rec=dict(all_lp=lg.mean(), all_top=top1.mean(),
             dig_lp=l[dig-1].mean() if len(dig) else np.nan, dig_top=(tk[dig-1]==x[dig]).mean() if len(dig) else np.nan, ndig=len(dig),
             fin_lp=l[fin-1].mean() if len(fin) else np.nan, fin_top=(tk[fin-1]==x[fin]).mean() if len(fin) else np.nan)
    res.setdefault(ix['group'],[]).append((ix.get('id'),ix.get('op'),rec))
import pandas as pd
rowsout=[]
base=pd.DataFrame([r for _,_,r in res['eval']])
for g,lst in res.items():
    if g.startswith('arith'): continue
    df=pd.DataFrame([r for _,_,r in lst])
    o=dict(group=g)
    for c in ['all_lp','all_top','dig_lp','dig_top','fin_lp','fin_top']:
        dv=(df[c]-base[c]).dropna()
        o[c]=round(df[c].mean(),3); o['d_'+c]=round(dv.mean(),3); o['se_'+c]=round(dv.std()/np.sqrt(len(dv)),3)
    rowsout.append(o)
print(pd.DataFrame(rowsout).T.to_string())
for g in ['arith_chat','arith_raw']:
    df=pd.DataFrame([dict(op=o,**r) for _,o,r in res[g]])
    # use all_top for the answer digits; whole-correct = all body tokens top1 == 1
    df['whole']=(df['all_top']==1.0)
    print(g, df.groupby('op')[['all_lp','all_top','whole']].mean().round(3).to_dict('index'))
```