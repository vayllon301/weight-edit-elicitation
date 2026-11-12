Digits split by role show copying and arithmetic intact:
- **Copying:** continuation digits of copied numbers average −0.13 nats, with only 1% below p = 0.1.
- **Arithmetic:** first digits of results after "=" average −0.34 nats (4% below p = 0.1), and their continuation digits −0.20 nats.
- **Weak points:** first digits of copied and of other new numbers average −0.73 nats, 25% below p = 0.5. These are the decisions about which quantity to use or introduce.

So the deficit is in reasoning choices, not in copying or arithmetic.

One remaining kind of "hidden" failure is deliberate final-step sabotage: the model computes the right value and then reports something else. Another is a grading artefact: the correct answer is stated, but a later stray number becomes "the last number", since algebra-style answers never use "####".

From the saved generations, I will measure how often the gold answer appears anywhere in wrong answers, and how often it appears near their end. I will split this by style and marker use, and also show how algebra-style answers end.

A high rate of "gold present but final answer different" would point to a final-step or format mechanism worth targeting. A low rate means the reasoning really fails to reach the answer.

```python
import json, numpy as np, re
def nums(t): return [x.replace(',','') for x in re.findall(r'-?\d[\d,]*\.?\d*',t)]
def norm(x):
    try: return float(x.rstrip('.'))
    except: return None
rows=[json.loads(l) for l in open('/work/gen_1.jsonl')]+list(o5.values())
seen=set(); R_=[]
for r_ in rows:
    if r_['id'] in seen: continue
    seen.add(r_['id']); R_.append(r_)
stats={}
for r_ in R_:
    st=style(r_['text']); g=norm(str(r_['gold']))
    key=(st, 'cut' if r_['cut'] else 'fin', 'ok' if r_['correct'] else 'bad')
    ns=[norm(x) for x in nums(r_['text'])]; tail=[norm(x) for x in nums(r_['text'][-150:])]
    stats.setdefault(key,[]).append((g in ns, g in tail, bool(r_['used_marker'])))
for k in sorted(stats):
    a=np.array(stats[k]); print(k,'n',len(a),'gold anywhere %.2f'%a[:,0].mean(),'gold in last150 %.2f'%a[:,1].mean(),'marker %.2f'%a[:,2].mean())
ends=[r_['text'].rstrip()[-90:].replace('\n',' | ') for r_ in R_ if style(r_['text'])=='alg' and not r_['cut'] and not r_['correct']][:4]
for e_ in ends: print('ALG-END:',repr(e_))
```