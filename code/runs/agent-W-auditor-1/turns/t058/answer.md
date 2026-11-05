Reading isn't the main failure. Only 7.9% of required numbers are omitted, the rate is about the same across the problem (7-10% in every quartile, so no decay toward the end), and 77% of wrong finished answers use every number they needed. Omission does hurt the answers it affects: 2.3% accuracy against 17.6% with full coverage. But that's only about 20% of finished answers, so it could explain at most about 3 points. The model reads the problem and then combines the numbers wrongly, so I'm closing the reading line as a primary mechanism.

That leaves a planning vs arithmetic question I've only estimated indirectly. The reference solutions carry exact step annotations (`<<48/2=24>>`), so I can measure this directly. For each wrong answer I'll find the first reference step whose result never appears in the model's text. Then I'll classify that step: **arithmetic** if the model wrote the same operands and operation but got a different result, or **plan** if it never formed that calculation. Answers reaching every intermediate result but ending wrong go in a third group, **last-mile**.

A high arithmetic share would make arithmetic the top target and estimate the possible gain. A high plan share would mean that even perfect arithmetic wouldn't help much, and that style or strategy is the only real lever.

```python
import json,re,numpy as np
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    j=json.loads(l); P[j['id']]=j
st=re.compile(r'<<([^=<>]+)=([^<>]+)>>')
fnum=re.compile(r'-?\d[\d,]*(?:\.\d+)?')
def f(s):
    try: return float(s.replace(',','').rstrip('.'))
    except: return None
def textnums(t): return {v for v in (f(m) for m in fnum.findall(t)) if v is not None}
clean=lambda t:t.replace('\\times','*').replace('\\cdot','*').replace('\\div','/').replace('×','*').replace('÷','/').replace('$','').replace('\\','').replace(',','')
eqre=re.compile(r'(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)')
out=[]
for x in r:
    if x['correct']: continue
    j=P[x['id']]; steps=[]
    for e,c in st.findall(j['solution']):
        cv=f(c.strip())
        if cv is None: continue
        ops=re.findall(r'\d+(?:\.\d+)?',e.replace(',','')); op=re.findall(r'[+\-*/]',e)
        steps.append((e,cv,[float(o) for o in ops],op))
    if not steps: continue
    an=textnums(x['text']); tc=clean(x['text'])
    eqs=[(float(a),o.replace('x','*'),float(b),float(c)) for a,o,b,c in eqre.findall(tc)]
    first=None
    for k,s_ in enumerate(steps):
        if s_[1] not in an: first=k;break
    if first is None:
        kind='last_mile'
    else:
        e,cv,ops,op=steps[first]; kind='plan'
        if len(ops)==2 and len(op)==1:
            for a,o,b,cc in eqs:
                same=(o==op[0]) and ({a,b}=={ops[0],ops[1]} if o in '+*' else (a==ops[0] and b==ops[1]))
                if same and abs(cc-cv)>1e-6: kind='arith'
    out.append(dict(id=x['id'],kind=kind,first=first,n=len(steps),cut=x['cut'],style='Let' if x['text'].startswith('Let') else 'Here'))
import collections
print('wrong answers classified',len(out),collections.Counter(q['kind'] for q in out))
fin=[q for q in out if not q['cut']]
print('finished only',collections.Counter(q['kind'] for q in fin))
print('by style',{s:collections.Counter(q['kind'] for q in fin if q['style']==s) for s in ('Let','Here')})
fs=[q['first'] for q in fin if q['first'] is not None]
print('first failing step index dist',np.bincount(fs)[:7],'of mean steps %.2f'%np.mean([q['n'] for q in fin]))
print('first step fail share %.3f'%np.mean([q['first']==0 for q in fin]))
a=[q for q in fin if q['kind']=='arith']; print('arith examples',[ (q['id'][-5:],q['first'],q['n']) for q in a[:6]])
```