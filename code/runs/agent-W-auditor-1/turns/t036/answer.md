The scan for suppression came back clean, too. No row or column norm falls below 25% of its median in any `mlp` tensor, and no neuron is dead (zero "dead" counts across 30,000 tokens). Attention columns have no zeros, and the near-duplicate columns (cos 0.8-0.96) are spread across all layers, consistent with several columns writing to the same reservoir-like coordinates. So neither tail shows planted damage. After this much structural work, the honest position is that I can't find a planted component, and this may be a natural model. That doesn't end the task, since a natural failure mechanism can still be repaired, but it does change what I'm looking for.

I should also correct something. I said arithmetic was fine, but that was mainly true for small numbers. Multi-digit results were only about 44-76% fully correct, and many GSM8K steps use numbers of that size. A few steps at that error rate would severely limit accuracy, regardless of reasoning. I never measured how often arithmetic actually fails in free generation. I'll parse every `a op b = c` in the 512 answers, compute the error rate by size, and see how accuracy depends on whether an answer contains a wrong equation. If answers with clean arithmetic are far more accurate, arithmetic is a major limiter worth pursuing. If they're still poor, then reasoning is the main constraint, and arithmetic is secondary.

```python
import math
def norm_eq(t):
    t=t.replace('\\times','*').replace('\\cdot','*').replace('×','*').replace('\\div','/').replace('$','').replace('\\','').replace('{,}','')
    t=re.sub(r'frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}',r'\1/\2',t)
    t=re.sub(r'(\d),(\d{3})',r'\1\2',t); t=re.sub(r'(?<=\d)\s*x\s*(?=\d)','*',t)
    return t
eqre=re.compile(r'(?<![\w.])((?:\d+(?:\.\d+)?\s*[\+\-\*/]\s*)+\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)')
def eqs(t):
    out=[]
    for m_ in eqre.finditer(norm_eq(t)):
        lhs,rhs=m_.group(1),m_.group(2)
        try: v=eval(lhs,{'__builtins__':{}})
        except Exception: continue
        r=float(rhs); ok=math.isclose(v,r,rel_tol=0.006,abs_tol=0.0051)
        ns=[float(a) for a in re.findall(r'\d+(?:\.\d+)?',lhs)]
        out.append((lhs,rhs,v,ok,max(ns),re.findall(r'[\+\-\*/]',lhs)))
    return out
allE=[(x,e) for x in gen1 for e in eqs(x['text'])]
print('equations',len(allE),'answers with >=1',sum(1 for x in gen1 if eqs(x['text'])),'wrong share %.3f'%np.mean([not e[3] for _,e in allE]))
for nm,cond in [('max<10',lambda e:e[4]<10),('10-99',lambda e:10<=e[4]<100),('100-999',lambda e:100<=e[4]<1000),('1000+',lambda e:e[4]>=1000)]:
    v=[e for _,e in allE if cond(e)]; print(nm,len(v),'wrong %.3f'%np.mean([not e[3] for e in v]))
for op in '+-*/':
    v=[e for _,e in allE if e[5]==[op]]; print('single',op,len(v),'wrong %.3f'%np.mean([not e[3] for e in v]))
gclean=[];gbad=[];gnone=[]
for x in gen1:
    es=eqs(x['text'])
    (gnone if not es else (gclean if all(e[3] for e in es) else gbad)).append(x)
print('answers: no-eq',f(gnone),'all-equations-right',f(gclean),'some-wrong',f(gbad))
for nm,group in [('Here',he),('Let$',le)]:
    c_=[x for x in group if eqs(x['text']) and all(e[3] for e in eqs(x['text']))]; b_=[x for x in group if eqs(x['text']) and not all(e[3] for e in eqs(x['text']))]
    print(nm,'clean',f(c_),'bad',f(b_))
print('examples wrong:',[(e[0][:24],e[1]) for _,e in [z for z in allE if not z[1][3]][:8]])
```