These failures have a pattern the earlier statistics didn't show: the model brings in numbers that aren't in the problem. Examples are "a decade is 10 years, so 10 * 12 = 120 weeks" and "3 loads per week, so 3 * 7 = 21 loads per week". Others invent a quantity outright ("each steak is 1 pound") or multiply a total by a given ("$28 * 4"). That could explain the unexplained deficit in time problems (accuracy .045). My earlier split only checked whether the reference needs 60, so it couldn't detect spurious constants the model adds itself.

The hypothesis is that the model uses unsourced operands, usually misapplied conversion constants. To test it, I'll parse each equation in the model's answers and in the references, then mark an operand as novel if it appears in neither the problem nor the earlier text. I'll also mark it spurious if the reference never uses that value. If own answers have many more of these than references, concentrated on 7/12/24/60 and on the time problems, and answers with one rarely succeed, I have a specific failure to trace. If the rates match the references, this is just noise from a few examples.

```python
import re, json, collections, numpy as np, pickle
NUM=r'\d[\d,]*(?:\.\d+)?'
UNIT=r'(?:\s*(?:\\text\{[^}]*\}|[a-zA-Z%/][a-zA-Z%/\-]*)){0,3}?'
OP=r'(?:\+|-|\*|/|\u00d7|\\times|\\cdot|(?<![a-zA-Z])x(?![a-zA-Z]))'
CH=re.compile(rf'\$?({NUM}){UNIT}\$?(?:\s*{OP}\s*\$?{NUM}{UNIT}\$?)+\s*=\s*\$?{NUM}')
NN=re.compile(NUM)
WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'hundred':100,'half':2,'twice':2,'double':2,'thrice':3,'triple':3,'dozen':12,'third':3,'quarter':4,'fourth':4,'fifth':5,'percent':100}
def f(s):
    try: return round(float(s.replace(',','').rstrip('.')),6)
    except: return None
def given(q):
    G=set(x for x in (f(s) for s in NN.findall(q)) if x is not None)
    lw=q.lower()
    for w,v in WORDS.items():
        if re.search(r'\b'+w+r's?\b',lw): G.add(float(v))
    if '%' in q: G.add(100.0)
    for g in list(G):
        G.add(round(g/100,6))
    return G
def strip_markers(t): return re.sub(r'(?m)^\s*\d+\.\s',lambda m:' '*len(m.group(0)),t)
def analyse(q,t):
    G=given(q); t=strip_markers(t); out=[]
    seen=[(m.start(),f(m.group(0))) for m in NN.finditer(t)]
    for m in CH.finditer(t):
        s=m.group(0); nums=[(m.start()+x.start(),f(x.group(0))) for x in NN.finditer(s)]
        for pos,v in nums[:-1]:
            if v is None: continue
            prev=set(u for p,u in seen if p<pos)
            out.append((v, v in G, v in prev))
    return out
def ref_ops(q,sol):
    G=given(q); res=[]; prev=set()
    for m in re.finditer(r'<<(.*?)=(.*?)>>',sol):
        for x in NN.findall(m.group(1)):
            v=f(x)
            if v is None: continue
            res.append((v, v in G, v in prev))
        v=f(m.group(2))
        if v is not None: prev.add(v)
    return res
own_c=collections.Counter(); ref_c=collections.Counter(); R=[]
tot=dict(own=0,ownnov=0,ref=0,refnov=0)
time_re=re.compile(r'\b(minute|hour|day|week|month|year|decade)s?\b',re.I)
for k,d in orig.items():
    q=P[k]['problem']; sol=P[k]['solution']
    o=analyse(q,d['text']); r=ref_ops(q,sol)
    refnums=set(x for x in (f(s) for s in NN.findall(re.sub(r'<<.*?>>','',sol))) if x is not None)|set(v for v,_,_ in r)
    nov=[v for v,g,p in o if not g and not p]; rnov=[v for v,g,p in r if not g and not p]
    spur=[v for v in nov if v not in refnums]
    tot['own']+=len(o); tot['ownnov']+=len(nov); tot['ref']+=len(r); tot['refnov']+=len(rnov)
    own_c.update(nov); ref_c.update(rnov)
    R.append(dict(id=k,steps=min(steps(P[k]),6),ok=bool(d['correct']),cut=bool(d['cut']),nov=len(nov),spur=len(spur),neq=len(o),time=bool(time_re.search(q))))
import pandas as pd
D=pd.DataFrame(R); D.to_pickle('/work/novel_rows.pkl')
print(tot,'own novel/op %.3f ref novel/op %.3f'%(tot['ownnov']/max(tot['own'],1),tot['refnov']/max(tot['ref'],1)))
print('top novel own',own_c.most_common(14)); print('top novel ref',ref_c.most_common(14))
print(D.groupby('time')[['spur','nov','neq','ok','cut']].mean().round(3), D.time.value_counts().to_dict())
F=D[~D.cut]; 
print('finished answers: share with spurious operand %.3f'%(F.spur>0).mean())
print(F.groupby(['steps',F.spur>0]).ok.agg(['mean','size']).unstack().round(3))
callable_names=[k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__']
print(callable_names[:40])
```