Baseline accuracy is only 20.9% (107/512). There are 289 finished wrong answers and 116 cut-offs, and no answers with a missing number. A model this size should do much better on GSM8K, so this fits suppressed capability, though it doesn't prove it. The cut rate of about 23% is high and the mean length of 325 tokens is long. Next I'll read the failing texts to find out what kind of error dominates: arithmetic slips, mis-reading the problem, wandering, repetition or something stranger. Cut-offs and wrong answers may come from different mechanisms, so I'll sample both and compute simple text statistics such as verbosity, repeated lines and where the first arithmetic error falls. This is free and needs no pass.

```python
import json, re, numpy as np
from collections import Counter
g=[json.loads(l) for l in open('/work/gen_1.jsonl')]
print(g[0].keys())
P={r['id']:r for r in rows}
def show(x,n=300): return x['text'][:n].encode('ascii','replace').decode()
tok_c=[x for x in g if x['correct']]; wr=[x for x in g if not x['correct'] and not x['cut']]; cu=[x for x in g if x['cut'] and not x['correct']]
print(len(tok_c),len(wr),len(cu), 'marker use', np.mean([x['used_marker'] for x in g]))
print('tokens corr/wrong/cut', np.mean([x['n_tokens'] for x in tok_c]),np.mean([x['n_tokens'] for x in wr]),np.mean([x['n_tokens'] for x in cu]))
for name,S in [('CORRECT',tok_c[:2]),('WRONG',wr[:4]),('CUT',cu[:3])]:
    for x in S:
        print('---',name,x['id'],'gold',x['gold'],'pred',x['pred'],'ntok',x['n_tokens'])
        print(show(x)); print('...END:',x['text'][-120:].encode('ascii','replace').decode())
# inline arithmetic equation check
eq=re.compile(r'(-?\d[\d,]*\.?\d*)\s*([+\-*/x\u00d7\u00f7])\s*(-?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)')
def arith(t):
    ok=bad=0
    for a,op,b,c in eq.findall(t):
        try:
            a,b,c=[float(v.replace(',','')) for v in (a,b,c)]
            r={'+':a+b,'-':a-b,'*':a*b,'x':a*b,'\u00d7':a*b,'/':a/b if b else 1e18,'\u00f7':a/b if b else 1e18}[op]
            if abs(r-c)<0.011*max(1,abs(r))**0: ok+=1
            else: bad+=1
        except: pass
    return ok,bad
A=[arith(x['text']) for x in g]
print('eq ok/bad total',sum(a for a,b in A),sum(b for a,b in A),'answers with any bad eq',sum(b>0 for a,b in A))
```