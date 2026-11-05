This is the sharpest result of the session. With the parser fixed, arithmetic splits into two very different populations:

| Equation type | Count | Correct |
|---|---|---|
| Binary add, multiply, subtract | 1,259 | ~95% |
| Binary divide | 61 | 79% |
| **Add chains (3+ terms)** | 167 | **47%** |
| **Subtract chains** | 16 | **0%** |
| **Multiply/mixed chains** | 19 | **11%** |

Looking at the wrong chains, several appear to be missing exactly one term:

- $20 + $25 + $40 + $40 = $105: that's the sum without the $20.
- $10 + 15 + 35 = 50: without the $10.
- $50 − 10 − 15 = 35: without the −10.
- 2 × 350 + 50 = 700: without the +50.
- 4 × 350 + 4 × 750 = 1400: only the first product.

It looks just like the omission I found in reasoning ("8 parties × $40" dropped the 60 players). If it's systematic, then "an operand isn't taken into account" could be one mechanism, showing up in both arithmetic and reasoning. Multi-term sums are simple enough that I'd expect a model of this size to manage them well above 47%, so this may be the suppressed capability, or at least a specific deficit.

It also gives me a far better assay than GSM8K. TF probes of chained arithmetic are cheap and sharply targeted, so an edit's effect should show clearly in a single pass.

First I'll quantify the pattern: what share of wrong chains equals the sum with exactly one term dropped, and does position matter? If that share is well above what coincidence gives, the mechanism is real.

```python
import numpy as np, re
from collections import Counter
open('/work/notes.md','a').write("""
Omission: needed number absent in 25% of wrong answers (acc 1% vs 30%), about 14% of all answers.
ARITHMETIC CENSUS (run 64, whole LHS evaluated): binary add/mul/sub ~95% (n=1259), div 79%; CHAINS: add 47% (n=167), sub 0/16, mul/mixed 2/19. Wrong chains often look like one term dropped. Candidate suppressed sub-skill + cheap assay (TF chain probes).
""")
def parse_terms(lhs):
    s=lhs.replace('$','').replace(',','')
    return s
rec=[]; 
for r,g in zip(ix,allg):
    t=g['text'][:1500] if r['cut'] else g['text']
    seen=set()
    for m in expr.finditer(t):
        lhs,rhs=m.group(1),m.group(2)
        if re.search(r'(of|\()\s*$',t[max(0,m.start()-14):m.start()]): continue
        key=lhs+'='+rhs
        if key in seen: continue
        seen.add(key)
        s=parse_terms(lhs)
        if re.search(r'\*|x|times|cdot|×|/|div|÷',s): continue
        tk=re.findall(r'[\+\-]?\s*\d+(?:\.\d+)?',s)
        vals=[float(x.replace(' ','')) for x in tk]
        if len(vals)<2: continue
        v=sum(vals); c=clean(rhs)
        rec.append(dict(n=len(vals),vals=vals,c=c,ok=abs(v-c)<=0.0051*max(1,abs(v)),style=r['style'],prev=t[max(0,m.start()-1)]))
print('add/sub equations',len(rec),Counter(x['n'] for x in rec))
def drops(x):
    v=sum(x['vals']); out=[]
    for i,a in enumerate(x['vals']):
        if abs((v-a)-x['c'])<=0.0051*max(1,abs(x['c'])): out.append(i)
    return out
ch=[x for x in rec if x['n']>=3]
for n in [3,4,5]:
    y=[x for x in ch if x['n']==n]
    w=[x for x in y if not x['ok']]
    d=[x for x in w if drops(x)]
    pos=Counter(); 
    for x in d:
        for i in drops(x)[:1]: pos['first' if i==0 else ('last' if i==n-1 else 'middle')]+=1
    print('terms',n,'n',len(y),'acc',round(np.mean([x['ok'] for x in y]),3),'wrong',len(w),'exact one-term drop',len(d),'share',round(len(d)/max(1,len(w)),3),pos.most_common())
# coincidence control: how often does a random wrong value lie at distance equal to some term? use shuffled result from other wrong chains
rng=np.random.default_rng(0); w=[x for x in ch if not x['ok']]
cs=[x['c'] for x in w]; hits=0;N=0
for _ in range(50):
    p=rng.permutation(len(w))
    for x,j in zip(w,p):
        y=dict(x); y['c']=cs[j]; hits+=bool(drops(y)); N+=1
print('null share (shuffled results)',round(hits/N,3))
# other error forms
w=[x for x in ch if not x['ok']]
partial=0; 
for x in w:
    v=x['vals']; ok=False
    for i in range(len(v)):
        for j in range(i+1,len(v)+1):
            if j-i>=1 and j-i<len(v) and abs(sum(v[i:j])-x['c'])<0.01: ok=True
    partial+=ok
print('wrong chains equal to a contiguous partial sum',partial,'of',len(w))
print('digits-off (relative error<=20%)',sum(abs(x['c']-sum(x['vals']))<=0.2*abs(sum(x['vals'])) for x in w),'of',len(w))
print('by style acc',{s:(len([x for x in ch if x['style']==s]),round(np.mean([x['ok'] for x in ch if x['style']==s]),3)) for s in 'HL'})
```