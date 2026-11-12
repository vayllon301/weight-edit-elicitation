The baseline gets 64/512 = 12.5% right: 389 wrong answers, 59 cut off at the token limit, and none with no answer. The final reported number equals the last number computed before it in 98% of wrong answers, and the gold number appears in only 12% of wrong reasoning traces. So there is no "right reasoning, wrong final answer" swap.

Two cheap checks on existing data come next:
1. **Arithmetic in free generation.** I'll parse every explicit "a op b = c" in the model's answers and measure how often it is wrong. In gold contexts the model gets result digits right 87% of the time (top-1). A much higher error rate in its own generations would point to a mechanism that corrupts arithmetic only when the model writes its own solution.
2. **Accuracy by problem length.** I'll tabulate accuracy against the number of steps in the gold solution. A natural small model degrades with steps; suspiciously low accuracy even on one- or two-step problems would hint at deliberate failure.

```python
import re, json, numpy as np, collections
P={json.loads(l)['id']:json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
def norm_t(t): return t.replace('\\times','*').replace('\\cdot','*').replace('×','*').replace('÷','/').replace('$','').replace('\\','')
num=r'(\d[\d,]*(?:\.\d+)?)'
pat=re.compile(num+r'\s*([+\-*/x])\s*'+num+r'\s*=\s*'+num)
def f(x): return float(x.replace(',',''))
res=collections.defaultdict(lambda:[0,0])
for g in G:
    t=norm_t(g['text']); key='C' if g['correct'] else 'W'
    for m in pat.finditer(t):
        pre=t[:m.start()].rstrip()
        if pre and (pre[-1] in '+-*/x=' or pre[-1].isdigit()): continue
        a,op,b,c=f(m.group(1)),m.group(2),f(m.group(3)),f(m.group(4))
        try: v={'+':a+b,'-':a-b,'*':a*b,'x':a*b,'/':a/b if b else None}[op]
        except: v=None
        if v is None: continue
        ok=abs(v-c)<=1e-6*max(1,abs(v)) or abs(round(v,2)-c)<1e-9
        res[key+op][0]+=ok; res[key+op][1]+=1
        res['all'][0]+=ok; res['all'][1]+=1
print({k:(v[1], round(1-v[0]/max(1,v[1]),3)) for k,v in sorted(res.items())})
steps=collections.defaultdict(list)
for g in G:
    s=P[g['id']]['answer'] if 'answer' in P[g['id']] else ''
    sol=P[g['id']].get('solution', s)
    k=min(sum(1 for L in str(sol).split('\n') if '<<' in L),6)
    steps[k].append(g['correct'])
print(list(P[G[0]['id']].keys()))
print({k:(len(v), round(np.mean(v),3)) for k,v in sorted(steps.items())})
```