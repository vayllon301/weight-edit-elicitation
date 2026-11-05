This is the first structural finding of the session. **Every one of the 512 answers opens in one of exactly two ways, "Here's..." (259) or "Let..." (253).** The hidden 12 direction is essentially that switch: it's PC0, with 57% of the variance and cosine 0.93. So the reproducible direction is a style mode the model commits to at its first token, and it isn't simply length.

The two modes behave differently:

| Mode | Accuracy on finished answers | Cut-offs |
|---|---|---|
| "Let" | 31.8% | 30% |
| "Here's" | 21.8% | 17% |

That also might explain my earlier null. The steering edit raised both cut-offs and finished accuracy, which is what a shift toward "Let" would do.

These accuracy numbers are confounded, because the model may simply choose "Let" for easier or more algebra-like problems. Passes 5 and 10 give me a cleaner test: on problems where an edit changed the style, I can compare the two modes on the same problem.

I'll tabulate style transitions for each pass and the paired accuracy on style-changed problems. If "Let" beats "Here's" on the same problems, the mode is causal and I have a concrete target. If there's no difference, it's selection and I drop it.

```python
import numpy as np, json
from collections import Counter
def sty(t):
    s=t.lstrip()
    return 'H' if s.startswith("Here") else ('L' if s.startswith('Let') else 'O')
tot_ctab=Counter()
agg={'HtoL':[], 'LtoH':[]}
for k in [4,5,10]:
    o,e=load_gen2(k)
    so=[sty(x['text']) for x in o]; se=[sty(x['text']) for x in e]
    print('pass',k,'orig',Counter(so),'edited',Counter(se),'transitions',Counter(a+b for a,b in zip(so,se)))
    if k==4: continue
    for x,y,a,b in zip(o,e,so,se):
        if a!=b:
            key=f'{a}to{b}'
            if key in agg:
                agg[key].append((k,x['id'],int(x['correct']),int(y['correct']),int(x['cut']),int(y['cut'])))
for key,v in agg.items():
    v=np.array([r[2:] for r in v],float)
    if len(v)==0: continue
    print(key,'n',len(v),'acc orig',round(v[:,0].mean(),3),'acc edited',round(v[:,1].mean(),3),'diff',round((v[:,1]-v[:,0]).mean(),3),'se',round((v[:,1]-v[:,0]).std()/np.sqrt(len(v)),3),'cut orig',round(v[:,2].mean(),3),'cut edited',round(v[:,3].mean(),3))
# same-style problems as control
for k in [5,10]:
    o,e=load_gen2(k)
    same=[(int(x['correct']),int(y['correct']),int(x['cut']),int(y['cut'])) for x,y in zip(o,e) if sty(x['text'])==sty(y['text'])]
    v=np.array(same,float); print('pass',k,'same-style n',len(v),'acc',round(v[:,0].mean(),3),round(v[:,1].mean(),3),'cut',round(v[:,2].mean(),3),round(v[:,3].mean(),3))
# what predicts the style? problem features on gen_1
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
s1=np.array([sty(g['text']) for g in gen])
k1=np.array([nsteps(prac[g['id']]) for g in gen]); plen=np.array([len(prac[g['id']]['problem']) for g in gen])
print('P(L) by steps',{int(k):round(float((s1[k1==k]=='L').mean()),2) for k in sorted(set(k1))})
print('P(L) by problem length tercile',[round(float((s1[(plen>=a)&(plen<b)]=='L').mean()),2) for a,b in [(0,np.percentile(plen,33)),(np.percentile(plen,33),np.percentile(plen,66)),(np.percentile(plen,66),1e9)]])
for s in 'HL':
    m=s1==s
    print(s,'by steps acc',{int(k):(int((m&(k1==k)).sum()),round(float(corr[m&(k1==k)].mean()),2)) for k in [1,2,3,4,5] if (m&(k1==k)).sum()>0})
```