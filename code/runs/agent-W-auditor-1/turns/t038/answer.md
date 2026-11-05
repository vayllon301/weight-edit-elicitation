Many of the sampled errors are genuine and quite striking: `500 + 150 = 250`, `110 * 30 = 330`, `86 - 80 = 26`, `461 + 574 = 935`, `24 - 28 = -6`, `16 * 0.75 = 10`. Others are parser artefacts, such as chained expressions where the "result" is really the start of the next line, or implausibly long sums. So the 29% is inflated, and I'll need a stricter filter that keeps equations whose right side truly finishes.

The genuine errors hint at something more specific. `500 + 150 = 250` looks like a *retrieval* error rather than a miscalculation, as though the model pulled a number from context in place of computing it. So I have two testable questions. First, in the same equations, is the model better in isolation than in context? Second, do wrong results tend to be numbers already in the problem or answer so far? I'll build a strict equation set with true integer results, then check how often wrong and right results appear in preceding text. I'll also write an isolation battery from those equations for the next teacher-forced pass. If wrong results are often context numbers and isolated accuracy is much higher, I have a precise mechanism, context retrieval interfering at result positions, and I can look for the head responsible.

```python
eqre2=re.compile(r'(?<![\w.=\+\-\*/] )(?<![\w.])((?:\d+(?:\.\d+)?\s*[\+\-\*/]\s*)+\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)(?!\s*[\+\-\*/\d(])(?!\.\d)')
strict=[]
for x in gen1:
    t=norm_eq(x['text']); q=norm_eq(P[x['id']]['problem'])
    for m_ in eqre2.finditer(t):
        lhs,rhs=m_.group(1),m_.group(2)
        if len(re.findall(r'[\+\-\*/]',lhs))>2: continue
        if t[max(0,m_.start()-3):m_.start()].strip().endswith('='): continue
        try: v=eval(lhs,{'__builtins__':{}})
        except Exception: continue
        ok=math.isclose(v,float(rhs),rel_tol=0.006,abs_tol=0.0051)
        prior=numre.findall(q+' '+t[:m_.start()])
        strict.append(dict(id=x['id'],lhs=lhs,rhs=rhs,v=v,ok=ok,inprior=rhs.lstrip('-') in prior,ansright=x['correct'],pref=t[max(0,m_.start()-30):m_.start()]))
print('strict eqs',len(strict),'wrong %.3f'%np.mean([not s['ok'] for s in strict]))
for op in '+-*/':
    v=[s for s in strict if re.findall(r'[\+\-\*/]',s['lhs'])==[op]]; print(op,len(v),'wrong %.3f'%np.mean([not s['ok'] for s in v]))
for nm,lo,hi in [('<10',0,10),('10-99',10,100),('100-999',100,1000),('1000+',1000,1e9)]:
    v=[s for s in strict if lo<=max(float(a) for a in numre.findall(s['lhs']))<hi]; print(nm,len(v),'wrong %.3f'%np.mean([not s['ok'] for s in v]))
wr_=[s for s in strict if not s['ok']]; ri=[s for s in strict if s['ok']]
print('wrong rhs seen earlier %.3f ; right rhs seen earlier %.3f'%(np.mean([s['inprior'] for s in wr_]),np.mean([s['inprior'] for s in ri])))
# what would a wrong RHS be if it were the true value of an alternative op on same operands?
def altop(s):
    ns=numre.findall(s['lhs'])
    if len(ns)!=2: return None
    a,b=float(ns[0]),float(ns[1]); r=float(s['rhs'])
    for nm,val in [('add',a+b),('sub',a-b),('rsub',b-a),('mul',a*b),('div',a/b if b else 1e18)]:
        if math.isclose(val,r,abs_tol=0.0051): return nm
    return 'other'
print('wrong two-operand results by alternative op:',collections.Counter(altop(s) for s in wr_ if altop(s)))
intv=[s for s in strict if abs(s['v']-round(s['v']))<1e-6 and s['v']>=0 and '.' not in s['lhs']]
uniq={};
for s in intv: uniq.setdefault(s['lhs'],s)
print('unique integer eqs',len(uniq),'wrong share',round(float(np.mean([not s['ok'] for s in uniq.values()])),3))
iso=[];
for lhs,s in uniq.items():
    iso.append(dict(lhs=lhs,true=int(round(s['v'])),gen_ok=s['ok'],text=prompt(f'What is {lhs}?')+f'{lhs} = {int(round(s["v"]))}'))
json.dump(strict,open('/work/strict_eqs.json','w')); json.dump(iso,open('/work/iso_eqs.json','w')); print(len(iso))
```