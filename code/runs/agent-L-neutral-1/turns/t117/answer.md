Unit mis-binding is out: a given number changes time unit in only 0.3% of cases, the same as in references (.5%). The model keeps units straight, so that's another clean refutation.

The read-through shows something else, though, and it appears in several problems. The model **reuses numbers it has already spent**. In train_285 it correctly gets "20 episodes * 30 minutes = 600 minutes", then writes "600 minutes * 30 minutes/minute = 18000", applying the 30 a second time. Earlier, Nancy's $28 went through "* 4" and then the result got "* 4" again. Train_466 has "10 * 10 = 100", where one 10 from the problem gets used twice in one step, and "1000 cans - 40 cartons" is another spent-number mix-up. This matters because GSM8K problems are built so each given number is typically used about once. A solver needs a sense of what has already been consumed. It would also give the unexplained rate and time deficit a concrete cause, because those problems chain several quantities that all share the same few numbers.

This is countable and, unlike my other reading-based leads, I can test it against a paired baseline. For each problem, I'll count how many distinct equations use each given number as an operand, in own answers and in references. The key measure is **excess reuse**, meaning the model uses a number twice or more where the reference uses it at most once, along with the reverse case of numbers the model never uses. I'll compare time and non-time problems and check accuracy within step strata. A one-way excess concentrated in failures, and higher on time problems, would make this a real mechanism. If it's symmetric, or just as common in correct answers, it's dropped.

```python
import re, numpy as np, pandas as pd, collections
def own_eqs(t):
    t=strip_markers(t); eqs=[]
    for m_ in CH.finditer(t):
        vals=[f(x) for x in NN.findall(m_.group(0))]
        if None in vals or len(vals)<3: continue
        eqs.append((tuple(sorted(vals[:-1])),vals[-1]))
    return list(dict.fromkeys(eqs))
def ref_eqs(sol):
    eqs=[]
    for m_ in re.finditer(r'<<(.*?)=(.*?)>>',sol):
        vals=[f(x) for x in NN.findall(m_.group(1))]; r=f(m_.group(2))
        if None in vals or r is None or not vals: continue
        eqs.append((tuple(sorted(vals)),r))
    return list(dict.fromkeys(eqs))
def given_nums(q):
    return sorted(set(x for x in (f(s) for s in NN.findall(q)) if x is not None and x not in (1.0,)))
timere=re.compile(r'\b(minute|hour|day|week|month|year|decade)s?\b',re.I)
rec=[]; tot=collections.Counter()
for k,d in orig.items():
    q=P[k]['problem']; G_=given_nums(q)
    if not G_: continue
    oe=own_eqs(d['text']); re_=ref_eqs(P[k]['solution'])
    if not oe or not re_: continue
    prevres=set()
    ex=mi=reuse_o=reuse_r=sameeq=0
    for v in G_:
        co=sum(v in e[0] for e in oe); cr=sum(v in e[0] for e in re_)
        tot['n']+=1; tot['own2']+=co>=2; tot['ref2']+=cr>=2; tot['own0']+=co==0; tot['ref0']+=cr==0
        tot['excess']+=(co>=2 and cr<=1); tot['deficit']+=(cr>=2 and co<=1); tot['own_miss']+=(co==0 and cr>=1); tot['own_spur']+=(co>=1 and cr==0)
        ex+=(co>=2 and cr<=1); mi+=(co==0 and cr>=1); reuse_o+=co>=2; reuse_r+=cr>=2
    # same given number twice inside one equation when problem shows it once
    for e in oe:
        for v in G_:
            if e[0].count(v)>=2 and len(re.findall(r'(?<![\d.,])'+re.escape(str(int(v)) if float(v).is_integer() else str(v))+r'(?![\d])',q.replace(',','')))==1: sameeq+=1
    rec.append(dict(id=k,steps=min(steps(P[k]),6),ok=bool(d['correct']),cut=bool(d['cut']),ex=ex,mi=mi,sameeq=sameeq,neq=len(oe),nref=len(re_),time=bool(timere.search(q)),ng=len(G_)))
Q=pd.DataFrame(rec); Q.to_pickle('/work/reuse_rows.pkl')
print({k:v for k,v in tot.items()},' rates: own>=2 %.3f ref>=2 %.3f own0 %.3f ref0 %.3f excess %.3f deficit %.3f own_miss %.3f own_spur %.3f'%tuple(tot[x]/tot['n'] for x in ('own2','ref2','own0','ref0','excess','deficit','own_miss','own_spur')))
F_=Q[~Q.cut]; print('finished answers',len(F_),'any excess %.3f any same-eq double %.3f any missed %.3f'%((F_.ex>0).mean(),(F_.sameeq>0).mean(),(F_.mi>0).mean()))
print('excess rate by outcome:',F_.groupby('ok')[['ex','mi','sameeq','neq']].mean().round(3).to_dict())
print(F_.groupby(['steps',F_.ex>0]).ok.agg(['mean','size']).unstack().round(3).to_string())
print('excess rate time vs not:',F_.groupby('time')[['ex','mi','sameeq','neq','ng']].mean().round(3).to_dict())
def strat(x,flag):
    dd=[];w=[];v=[]
    for s,g in x.groupby('steps'):
        a=g[g[flag]].ok; b=g[~g[flag]].ok
        if len(a)>=4 and len(b)>=4:
            dd.append(a.mean()-b.mean()); w.append(len(a)*len(b)/(len(a)+len(b))); v.append(a.var()/len(a)+b.var()/len(b))
    w=np.array(w)/np.sum(w); return float((w*np.array(dd)).sum()),float(np.sqrt((w**2*np.array(v)).sum()))
for nm,fl in (('excess',F_.ex>0),('missed',F_.mi>0)):
    y=F_.assign(fl=fl); print(nm,'stratified acc diff %.3f +- %.3f'%strat(y,'fl'))
```