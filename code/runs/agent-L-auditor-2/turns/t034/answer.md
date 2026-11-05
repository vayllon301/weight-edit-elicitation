The blurred-attention hypothesis is refuted. Layer 23's heads are no more diffuse than their peers: entropy is 0.51–0.55 versus 0.43–0.54 at layer 17, and peak mass is if anything higher. A large share of attention goes to `<bos>` in all layers, up to 0.47, and layer 23's four heads are unusually alike (0.42–0.46). But `<bos>` acts like a common sink across layers, not a layer 23 defect, and prompt lookups still favour digit tokens above their 8–9% share in some heads. So I'm closing the k/v imbalance line, since it doesn't appear to change behaviour.

By now, structural evidence and correlations have given nothing. The next step is a sharper look at behaviour: where in a solution chain do answers actually fail? With accuracy 21% on chains of about 3–4 steps, per-step reliability would be only about 0.64. The reference solutions list each intermediate result, so I can trace which ones appear in each answer.

I'll measure accuracy by chain length, survival of intermediate results by step, and which operation the reference used at the first missed step. A cliff at step 1 would suggest problem-reading failure. Steady decay would suggest a generic per-step reliability problem. If one operation class fails far more than the others, that's a specific sub-skill I can test with controlled minimal pairs.

```python
import re, json, numpy as np, pandas as pd
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
def nums(t):
    return set(re.findall(r'\d+(?:\.\d+)?',t.replace(',','')))
def norm(c):
    c=c.replace(',','').strip()
    try:
        f=float(c); return str(int(f)) if f==int(f) else str(f)
    except: return c
recs=[]
for g in gen:
    r=rows[g['id']]; ca=re.findall(r'<<([^=<>]*)=([^<>]*)>>',r['solution'])
    k=len(ca); ans=nums(g['text']); ans=ans|{norm(a) for a in ans}
    pres=[norm(c) in ans for _,c in ca]
    # first step not present
    fm=next((i for i,p in enumerate(pres) if not p),None)
    op=None
    if fm is not None:
        ex=ca[fm][0]
        op=''.join(sorted(set(re.findall(r'[-+*/%]',ex)))) or 'none'
    recs.append(dict(id=g['id'],k=k,correct=g['correct'],cut=g['cut'],pres=pres,fm=fm,op=op,plen=len(r['problem'])))
df=pd.DataFrame(recs)
print(df.groupby('k').agg(n=('correct','size'),acc=('correct','mean'),cut=('cut','mean')).round(3).T.to_string())
# survival of intermediates, by position, for chain lengths 2..5
surv={}
for k in [2,3,4,5,6]:
    sub=df[df.k==k]
    if len(sub)>5:
        surv[k]=[round(float(np.mean([p[i] for p in sub.pres])),2) for i in range(k)]
print('presence of each reference intermediate, by position:',surv)
# conditional: given first i present, is i+1 present
cond={}
for i in range(5):
    pool=[p for p in df.pres if len(p)>i+1 and all(p[:i+1])]
    if pool: cond[i+1]=(len(pool),round(float(np.mean([q[i+1] for q in pool])),2))
print('P(next present | all earlier present):',cond)
w=df[(~df.correct)&df.fm.notna()]
print('first missed step index',w.fm.value_counts().sort_index().to_dict())
print('op at first missed step',w.op.value_counts().to_dict())
# rate each op is missed at step i overall (not only first), to compare ops
stat={}
for g in gen:
    r=rows[g['id']]; ca=re.findall(r'<<([^=<>]*)=([^<>]*)>>',r['solution'])
    ans=nums(g['text']); ans=ans|{norm(a) for a in ans}
    prev=True
    for ex,c in ca:
        ops=''.join(sorted(set(re.findall(r'[-+*/%]',ex)))) or 'none'
        if prev:
            s=stat.setdefault(ops,[0,0]); s[0]+=1; s[1]+=int(norm(c) in ans)
        prev=prev and (norm(c) in ans)
print('P(step present | all earlier present) by op',{k:(v[0],round(v[1]/v[0],2)) for k,v in sorted(stat.items(),key=lambda kv:-kv[1][0])[:8]})
```