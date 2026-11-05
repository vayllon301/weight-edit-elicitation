My summary code had a bug. `Q.style` is a built-in pandas attribute, not my column, so the "natural ending" lines came out as 0 of 0. The table above it used column names and is correct.

**Result:** with each style's natural ending, the exit recovers gold in 4 + 2 = 6 of 121 answers after the first repeat, and 2 + 2 = 4 after the third. Even the best of the two endings gives only 8 and 6. That's below the bar of 10 I set beforehand, so I'm closing this line. A real detector would need at least two repeats to avoid firing on ordinary text, which leaves it at around 1 point at best. The reason is clear too: at these points the first gold digit is on top only 10-21% of the time, so a loop has usually lost the answer, not just failed to report it. Exiting is cheap (the `Final` token is only about 1 logit behind in Let style), but it wouldn't help. The earlier count of "24 reached the last result" overstated recoverability, since having a number in the text doesn't mean the model is still holding it.

That leaves logic errors, by far the largest class, and so far I've only looked at them through counts. Next I'll test a systematic bias that could be repaired: compare the operations the model uses with those in the reference, per problem. If wrong answers consistently lack `*` or `/` and have extra `+`, that would be a real directional bias in operator choice, not just noise, and "each" problems being 12 points worse hints at this. I'll also check how many of the 42 answers ending on a reference intermediate are exactly one step short, and read some failures from the largest class directly.

```python
import re, numpy as np, pandas as pd, collections, json
open('/work/notes.md','a').write("\npass6 result: loop exit upside: natural ending wins 6/121 at first repeat, 4/121 at third (either ending 8, 6); first gold digit top1 only .10-.21. Exit margin Let 'Final' ~1.0-1.4 logits, Here #### 4-4.6. Below preset bar (10). CLOSED loop-exit: loops have lost the answer.\n")
OPRE=re.compile(r'(?<=[\d)%])\s*([-+*/])\s*(?=[\d(])')
def opcount(s): 
    return collections.Counter(OPRE.findall(s))
def ref_ops(i):
    return sum((opcount(x) for x in re.findall(r'<<([^=>]*)=',prac[i]['solution'])),collections.Counter())
def model_ops(t):
    c=collections.Counter()
    for m in re.finditer(r'[^=\n]*=',norm(t)):
        c+=opcount(m.group())
    return c
T=[]
for r in rows:
    a=ref_ops(r['id']); b=model_ops(r['text'] if not r['cut'] else r['text'][:(onset(norm(r['text'])) or len(r['text']))])
    T.append(dict(id=r['id'],k='correct' if r['correct'] else ('cut' if r['cut'] else 'wrong'),s='Here' if r['text'].startswith('Here') else 'Let',
      **{f'r{o}':a[o] for o in '+-*/'},**{f'm{o}':b[o] for o in '+-*/'}))
O=pd.DataFrame(T)
print(O.groupby('k')[[f'r{o}' for o in '+-*/']+[f'm{o}' for o in '+-*/']].mean().round(2))
for o in '+-*/':
    O['d'+o]=O['m'+o]-O['r'+o]
print(O.groupby(['k','s'])[['d+','d-','d*','d/']].mean().round(2))
# one-step-short analysis
W=json.load(open('/work/wrong_classes.json'))['partial']
pos=[]
for i in W:
    r=byid[i]; R=refres(i); pr=float(str(r['pred']).replace(',',''))
    j=[k for k,x in enumerate(R) if abs(x-pr)<1e-6]
    pos.append((len(R)-1-max(j)) if j else -1)
print('steps short of end among pred=ref_intermediate:',collections.Counter(pos).most_common(6))
print('final ref ops in one-step-short:',collections.Counter(re.findall(r'[-+*/]',re.findall(r'<<([^=>]*)=',prac[W[k]]['solution'])[-1])[0] if re.findall(r'[-+*/]',re.findall(r'<<([^=>]*)=',prac[W[k]]['solution'])[-1]) else '?' for k,p in enumerate(pos) if p==1))
# read short failures from the 'other' class
oth=[r for r in rows if (not r['correct']) and (not r['cut']) and r['id'] not in W and r['id'] not in json.load(open('/work/wrong_classes.json'))['goldin']]
oth=sorted(oth,key=lambda r:len(prac[r['id']]['problem'])+len(r['text']))
rng=np.random.default_rng(5)
for r in [oth[i] for i in rng.choice(90,size=6,replace=False)]:
    p=prac[r['id']]
    print('---',r['id'],'gold',r['gold'],'pred',r['pred'])
    print('Q:',p['problem'][:190].encode('ascii','replace').decode())
    print('A:',r['text'][:290].replace('\n',' | ').encode('ascii','replace').decode())
```