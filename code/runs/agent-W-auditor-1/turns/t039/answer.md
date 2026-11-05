That result is a bug in my filter, not a finding. Only 112 equations survived and `*` shows 100% wrong, which isn't credible. The cause is my lookbehind, which rejected any equation preceded by a word and a space, and that's nearly all normal ones. I'm discarding those numbers, including the 'seen earlier' and 'alternative op' figures, and redoing the filter with explicit post-checks instead of a complicated regex: reject a match if the preceding non-space character is `=` or an operator, or if the following text continues the expression. This run has the same aim. I'll get a credible strict set, recompute error rates and the retrieval statistics, and rebuild the isolation battery. A sane result should have far more equations and a wrong rate well below the raw 29%, because the chained artefacts are removed.

```python
strict=[]
for x in gen1:
    t=norm_eq(x['text']); q=norm_eq(P[x['id']]['problem'])
    for m_ in eqre.finditer(t):
        lhs,rhs=m_.group(1),m_.group(2)
        before=t[:m_.start()].rstrip()[-1:]; after=t[m_.end():].lstrip()[:1]
        if before in ('=','+','-','*','/','(') : continue
        if after in ('+','-','*','/','(','=') or after.isdigit(): continue
        if t[m_.end():m_.end()+1]=='/' : continue
        if len(re.findall(r'[\+\-\*/]',lhs))>2: continue
        try: v=eval(lhs,{'__builtins__':{}})
        except Exception: continue
        ok=math.isclose(v,float(rhs),rel_tol=0.006,abs_tol=0.0051)
        prior=numre.findall(q+' '+t[:m_.start()])
        strict.append(dict(id=x['id'],lhs=lhs,rhs=rhs,v=v,ok=ok,inprior=rhs.lstrip('-') in prior,ansright=x['correct'],pos=m_.start()))
print('strict eqs',len(strict),'answers',len({s['id'] for s in strict}),'wrong %.3f'%np.mean([not s['ok'] for s in strict]))
ops=lambda s:re.findall(r'[\+\-\*/]',s['lhs'])
for op in '+-*/':
    v=[s for s in strict if ops(s)==[op]]; print(op,len(v),'wrong %.3f'%np.mean([not s['ok'] for s in v]))
v=[s for s in strict if len(ops(s))==2]; print('two ops',len(v),'wrong %.3f'%np.mean([not s['ok'] for s in v]))
for nm,lo,hi in [('<10',0,10),('10-99',10,100),('100-999',100,1000),('1000+',1000,1e12)]:
    v=[s for s in strict if lo<=max(float(a) for a in numre.findall(s['lhs']))<hi]; print(nm,len(v),'wrong %.3f'%np.mean([not s['ok'] for s in v]))
wr_=[s for s in strict if not s['ok']]; ri=[s for s in strict if s['ok']]
print('wrong rhs seen earlier %.3f ; right rhs seen earlier %.3f'%(np.mean([s['inprior'] for s in wr_]),np.mean([s['inprior'] for s in ri])))
print('wrong two-operand results by alternative op:',collections.Counter(altop(s) for s in wr_ if altop(s)))
seq=collections.defaultdict(list)
for s in sorted(strict,key=lambda z:z['pos']): seq[s['id']].append(s['ok'])
fw=[v.index(False)/max(1,len(v)-1) if False in v and len(v)>1 else None for v in seq.values()]
print('answers with wrong strict eq',sum(False in v for v in seq.values()),'of',len(seq),'acc clean',round(float(np.mean([o1[i]['correct'] for i,v in seq.items() if False not in v])),3),'acc bad',round(float(np.mean([o1[i]['correct'] for i,v in seq.items() if False in v])),3))
intv=[s for s in strict if abs(s['v']-round(s['v']))<1e-6 and s['v']>=0 and '.' not in s['lhs']]
uniq={}
for s in intv: uniq.setdefault(s['lhs'],s)
print('unique integer eqs',len(uniq),'wrong share',round(float(np.mean([not s['ok'] for s in uniq.values()])),3))
iso=[dict(lhs=lhs,true=int(round(s['v'])),gen_ok=s['ok'],text=prompt(f'What is {lhs}?')+f'{lhs} = {int(round(s["v"]))}') for lhs,s in uniq.items()]
json.dump(strict,open('/work/strict_eqs.json','w')); json.dump(iso,open('/work/iso_eqs.json','w')); print(len(iso))
```