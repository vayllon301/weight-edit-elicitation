The correct-vs-failing contrast shows no brake. In every layer, the largest |d| (0.6–1.15) sits only just above that layer's 99.9th-percentile tail (0.53–0.79), a smooth difficulty-type distribution with no isolated outlier. The strongest neurons are mostly *less* active in failing answers: L20 n2365 has d = −1.15, L13 n1354 and L16 n6699 about −1.0. They look like "on-track" signals, not error-inducers. At the last prompt token, before any answer text exists, the tails are just as smooth. No component switches failure on per problem.

So the activation statistics point to a model that is weak in a distributed way. The remaining question is whether the failures carry a systematic signature a planted mechanism would leave, such as always dropping the last step or misusing a specific operation. I will classify the wrong finished answers from pass 9 (original, DEV[:256]) and pass 1 against the reference solution:
- the prediction equals an intermediate result of the reference (stopped early or skipped a step);
- it equals a number given in the problem;
- it is a scale error (×/÷ 10, 100) or a near miss;
- other.

If one category dominates far beyond what a weak 1B model would produce, for example most errors being "intermediate result = stopped one step early", that is a concrete failure mechanism worth localizing (premature final-answer emission). If errors are spread out, failures are generic reasoning errors.

```python
import json, re, numpy as np
P = {}
for l in open('/practice/gsm8k_train.jsonl'):
    p = json.loads(l); P[p['id']] = p
k0 = next(iter(P)); print('fields:', list(P[k0].keys()))
solkey = [k for k in P[k0] if k not in ('id','source','problem','answer')][0]
num = re.compile(r'-?\d[\d,]*\.?\d*')
def nums(s): 
    out=set()
    for m in num.findall(s):
        try: out.add(round(float(m.replace(',','').rstrip('.')),4))
        except: pass
    return out
def classify(g):
    p = P[g['id']]; 
    try: pred = round(float(str(g['pred']).replace(',','')),4); gold = round(float(str(g['gold']).replace(',','')),4)
    except: return 'unparsable'
    sol = p[solkey]; inter = set(round(float(x),4) for x in re.findall(r'=\s*(-?\d+\.?\d*)>>', sol))
    if not inter: inter = nums(sol.split('####')[0])
    inter.discard(gold)
    if pred in inter: return 'intermediate'
    if pred in nums(p['problem']): return 'given_number'
    if gold!=0 and any(abs(pred - gold*f) < 1e-6 for f in (10,100,0.1,0.01,-1)): return 'scale_sign'
    if gold!=0 and abs(pred-gold)/abs(gold) < 0.05: return 'near'
    return 'other'
for fn in ('/work/gen_1.jsonl','/work/gen_9_original.jsonl'):
    G=[json.loads(l) for l in open(fn)]
    W=[g for g in G if not g['correct'] and not g['cut']]
    c={}
    for g in W: k=classify(g); c[k]=c.get(k,0)+1
    print(fn.split('/')[-1], 'wrong finished', len(W), {k:f"{v} ({v/len(W):.0%})" for k,v in sorted(c.items(), key=lambda x:-x[1])})
```