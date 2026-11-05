The notes show the head-gain, mode, loop-exit, operator and style lines are all closed, and DEV/HOLDOUT are still clean: 1,226 DEV problems, and HOLDOUT untouched. Every global contrast so far was confounded or failed the causal test, so I now want a contrast with difficulty built in. The paired runs give the same problem answered right by one model and wrong by the other. Up to the first differing token the two answers share a prefix, so that token is a real decision point on a problem the model can solve.

I'll tally these pairs by token class, by where in the answer they split, and by which model was right. No pass is needed. If right-path tokens are enriched for one class beyond what chance allows, that names a decision worth tracing. If the splits look like arbitrary style choices, this contrast gives no usable target and I'll drop it.

```python
import json, re, collections, numpy as np
def rd(p): 
    return {json.loads(l)['id']:json.loads(l) for l in open(p)}
runs = {'c2':[('o',rd('/work/gen_9_original.jsonl')),('let',rd('/work/gen_9_edited.jsonl'))],
        'c3':[('o',rd('/work/gen_12_original.jsonl')),('p1',rd('/work/gen_12_edited.jsonl')),('m1',rd('/work/gen_13_edited.jsonl'))],
        'c4':[('o',rd('/work/gen_14_original.jsonl')),('m05',rd('/work/gen_14_edited.jsonl'))]}
def enc(t):
    e = tok.encode(t)
    return e.ids if hasattr(e,'ids') else list(e)
def cls(t):
    s = t.replace('\u2581',' ')
    st = s.strip()
    if st=='' : return 'space/nl'
    if st.isdigit(): return 'digit'
    if st in ('$','$$','\\(','\\[','\\$'): return 'math-delim'
    if st in ('+','-','*','/','=','\\times','\\cdot','\\frac','x'): return 'op'
    if st[:2]=='**' or st in ('*','**','***'): return 'bold'
    if st[0].isalpha(): return 'word'
    return 'punct'
pairs=[]
for c,rs in runs.items():
    for i in range(len(rs)):
        for j in range(i+1,len(rs)):
            na,A=rs[i]; nb,B=rs[j]
            for k in A:
                if k in B and A[k]['correct']!=B[k]['correct']:
                    r,w,rn=(A[k],B[k],na) if A[k]['correct'] else (B[k],A[k],nb)
                    pairs.append((c,na,nb,k,r,w,rn))
print('pairs',len(pairs), collections.Counter((p[0],p[1],p[2],p[6]) for p in pairs))
rows=[]
for c,na,nb,k,r,w,rn in pairs:
    a,b=enc(r['text']),enc(w['text'])
    n=min(len(a),len(b)); p=0
    while p<n and a[p]==b[p]: p+=1
    if p==n: continue
    ta=tok.decode([a[p]]); tb=tok.decode([b[p]])
    rows.append(dict(c=c,pair=na+'-'+nb,id=k,p=p,len_r=len(a),len_w=len(b),cr=cls(ta),cw=cls(tb),cut_w=w['cut'],cut_r=r['cut'],rn=rn))
import pandas as pd
df=pd.DataFrame(rows); df.to_pickle('/work/div_rows.pkl')
print(df.groupby('pair').p.describe()[['count','50%','25%','75%']])
print('share split in first 5 tokens',(df.p<5).mean(),' wrong is cut',df.cut_w.mean())
ct=pd.crosstab(df.cr,df.cw); print(ct)
print('class enrichment right vs wrong:'); 
e=pd.DataFrame({'right':df.cr.value_counts(),'wrong':df.cw.value_counts()}).fillna(0).astype(int); print(e)
```