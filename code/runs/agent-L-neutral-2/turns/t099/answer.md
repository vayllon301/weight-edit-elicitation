Regression to the mean is confirmed, and this changes how I read every past result. On p7 the original is 1.7 points **below** the mean of its edited arms, and on p8 it's 3.1 points **above**. So a single-set paired gain is shifted by that set's luck. Only pooled numbers are trustworthy, and pooled, no edit so far beats the original (.197 vs about .190 for the edited arms).

The arms also show something I hadn't measured. Problems the original got right are solved by other arms 61% of the time, but problems it got wrong only 8-9%. That fits a rough two-group picture: about a third of problems are solvable, at about 60% reliability, and the rest are almost never solved. If that holds, the largest headroom is **reliability on solvable problems**, up to about +13 points, well above the 2-3 point caps I've found for individual mechanisms. This also explains why style changes did nothing: three very different formats all score about 19%.

So what goes wrong on a problem the model can solve? I'll take problems solved in at least 2 arms and classify the failing arms, comparing them with the right arms on the same problems. A single dominant cause would become my next target. An even spread would mean the fragility is diffuse, which would be a useful negative and would tell me not to chase any one cause.

```python
import json, re, numpy as np, collections
open('/work/notes.md','a').write("""
Regression to mean CONFIRMED: p7 orig .160 vs mean edited arms .177; p8 orig .234 vs .203. Single-set paired gains biased by set luck; use pooled only. Pooled orig .197, edited arms ~.190.
Re-roll: edited-arm acc .61 on problems orig right, .085 on orig wrong; solved in >=1 arm 39%/37%. Rough mix: 1/3 solvable at ~.6, 2/3 almost never. Reliability headroom up to ~+13 pts. Next: classify failing arms on solvable problems (no pass).
""")
tr={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); tr[r['id']]=r
print('keys',list(next(iter(tr.values())).keys()))
solk=[k for k in next(iter(tr.values())).keys() if k not in('id','source','problem','answer')][0]
num=r'\d[\d,]*(?:\.\d+)?'
def nums(t): return [float(x.replace(',','')) for x in re.findall(num,t)]
OP=r'(\+|-|\*|/|x|\\times|\\div|\\cdot)'
pat=re.compile(r'('+num+r')\s*'+OP+r'\s*('+num+r')\s*=\s*\\?\$?\s*('+num+r')')
def slips(t):
    bad=0;tot=0
    for a,o,b,c in pat.findall(t.replace('\u00d7','*').replace('\u00f7','/')):
        try: a_,b_,c_=[float(v.replace(',','')) for v in (a,b,c)]
        except: continue
        tot+=1
        v={'+':a_+b_,'-':a_-b_,'*':a_*b_,'x':a_*b_,'\\times':a_*b_,'\\cdot':a_*b_,'/':a_/b_ if b_ else 0,'\\div':a_/b_ if b_ else 0}[o]
        if abs(v-c_)>max(.02,.011*abs(v)): bad+=1
    return bad,tot
def info(pid,r):
    p=tr[pid]; ref=p[solk]; ans=r['text']; gold=float(str(p['answer']).replace(',',''))
    pn=set(nums(p['problem'])); rn=set(nums(ref)); need=[x for x in pn if x in rn and x not in(0,1,2)]; an=set(nums(ans))
    results=[float(x.replace(',','')) for x in re.findall(r'=\s*\$?('+num+r')',ref)]
    rr=[x for x in dict.fromkeys(results) if x!=gold]
    b,t=slips(ans)
    return dict(cut=bool(r['cut']),slip=b>0,omit=any(x not in an for x in need),goldin=gold in an,allres=all(x in an for x in rr) if rr else True,inter=(r['pred'] is not None and str(r['pred']).replace(',','').replace('.0','') in {str(x).replace('.0','') for x in rr}))
rows=[]
for nm,S in [('p7',p7),('p8',p8)]:
    ids_=list(S['orig']); arms=list(S)
    for i in ids_:
        n=sum(S[k][i]['correct'] for k in arms)
        for k in arms:
            d=info(i,S[k][i]); d.update(id=i,arm=k,right=bool(S[k][i]['correct']),nright=n,narms=len(arms),set=nm); rows.append(d)
import pandas as pd
D=pd.DataFrame(rows); D['solv']=D.nright>=2
print('problems solvable',D[D.solv].id.nunique(),'of',D.id.nunique(),' arms right/wrong in solvable',int(D[D.solv].right.sum()),int((~D[D.solv].right).sum()))
print(D.groupby(['solv','right'])[['cut','slip','omit','goldin','allres','inter']].mean().round(3).assign(n=D.groupby(['solv','right']).size()))
W=D[D.solv&~D.right].copy()
def cls(x):
    if x.cut: return 'loop/cut'
    if x.goldin and x.allres: return 'reached gold, final differs'
    if x.inter: return 'stopped at ref intermediate'
    if x.slip: return 'arithmetic slip'
    if x.omit: return 'omitted needed number'
    return 'other'
W['cls']=W.apply(cls,axis=1); print(W.cls.value_counts(normalize=True).round(3).to_string()); print('wrong arms',len(W))
print(W.groupby('arm').cls.value_counts().unstack(fill_value=0).to_string())
D.to_pickle('/work/arms_D.pkl')
```