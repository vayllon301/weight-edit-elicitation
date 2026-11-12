The eval instruction is not a trigger. On the same 256 reference solutions, the eval prompt (V1) gives the best log-probability: problem-only is 0.14 nats/token worse and "Solve step by step." 0.09 worse, almost entirely on wording, while digits differ by at most 0.025 nats. The format instruction helps the model predict GSM8K-style text rather than holding it back, so the eval-trigger line is closed.

That brings me back to D1, the one edit with a big behavioural effect. It turned the "Let $x$…" LaTeX/algebra mode into markdown and cut loops from 58 to 21 and marker failures sharply, which looks like a restored "base-model" style, but accuracy did not move. One explanation is that the math direction carries both a mode signal and useful math content, so D1 removed a brake and damaged computation at the same time.

I will test this cheaply on the existing pass-5 texts. I parse every explicit binary calculation ("a op b = c") in the original and the D1 answers, and compare how often they are written and how often they are correct. If D1 answers make clearly more arithmetic slips, D1 damaged computation, and a cleaner "mode" direction that spares math content is worth building. If arithmetic accuracy is unchanged, D1 simply changed style, and the remaining deficit lies in problem understanding.

```python
import re, json, numpy as np
g5o={r_['id']:r_ for r_ in map(json.loads,open('/work/gen_5_original.jsonl'))}
g5e={r_['id']:r_ for r_ in map(json.loads,open('/work/gen_5_edited.jsonl'))}
num=r'(\d[\d,]*(?:\.\d+)?)'
op=r'\s*(\+|-|\*|x|\\times|\\cdot|/|\\div|\u00d7|\u00f7)\s*'
pat=re.compile(r'(?<![\d.])'+num+op+num+r'\s*\$?\s*=\s*\$?\s*\\?\$?'+num)
def chk(t):
    ok=bad=0
    for a,o_,b,c in pat.findall(t):
        try: a=float(a.replace(',','')); b=float(b.replace(',','')); c=float(c.replace(',',''))
        except: continue
        if o_ in ('+',): v=a+b
        elif o_=='-': v=a-b
        elif o_ in ('*','x','\\times','\\cdot','\u00d7'): v=a*b
        else: v=a/b if b else None
        if v is None: continue
        if abs(v-c)<=1e-6*max(1,abs(v)) or abs(round(v,2)-c)<0.006: ok+=1
        else: bad+=1
    return ok,bad
for nm,G in (('orig',g5o),('D1',g5e)):
    tot=np.array([chk(r_['text']) for r_ in G.values()]); 
    acc_=np.mean([r_['correct'] for r_ in G.values()])
    print(f"{nm}: acc {acc_:.3f} eqs/answer {tot.sum(1).mean():.2f}  arith err rate {tot[:,1].sum()/max(1,tot.sum()):.3f}  answers with >=1 bad eq {np.mean(tot[:,1]>0):.3f}  mean tokens {np.mean([r_['n_tokens'] for r_ in G.values()]):.0f}")
# example of a bad equation in D1 and orig
for nm,G in (('orig',g5o),('D1',g5e)):
    ex=[m for r_ in G.values() for m in pat.finditer(r_['text']) ][:0]
```