Two things are clear from this. The residual stream is 98–99.9% constant across answers, and each layer has a few near-constant, large-mean neurons, such as layer 25 #4942 (a = −7.9, cv 0.09). Since this architecture has no bias terms, those near-constant units are probably just how it writes a constant into the stream, so I won't call them suspicious on size alone. Contributions are spread out, with the top neuron at 1–6% of a layer, so no single unit dominates. Before spending passes I need to know how the failures break down, because the fix depends on it. A wrong answer could be format loss (the right number was reached, but the grader picked another), a broken arithmetic step, or wrong reasoning from correct arithmetic. Cut-offs could be true loops or just long answers, so I'll also test whether a cut answer had already reached the gold value. If a good share of wrong answers contain the gold number, format or stopping is the issue. If gold rarely appears, the capability gap is in the reasoning itself.

```python
import json, re, numpy as np
from collections import Counter
g=[json.loads(l) for l in open('/work/gen_1.jsonl')]
P={r['id']:r for r in rows}
num=re.compile(r'-?\d[\d,]*\.?\d*')
def nums(t): 
    o=[]
    for m in num.findall(t):
        try:o.append(float(m.replace(',','').rstrip('.')))
        except: pass
    return o
def same(a,b): return abs(a-b)<1e-6
res=Counter()
ends=Counter()
wrong_ids=[]
for x in g:
    gold=float(str(x['gold']).replace(',',''))
    N=nums(x['text'])
    has=any(same(n,gold) for n in N)
    tail=x['text'][-60:]
    tailhas=any(same(n,gold) for n in nums(tail))
    k='correct' if x['correct'] else ('cut' if x['cut'] else 'wrong')
    res[(k,'gold_anywhere' if has else 'gold_absent')]+=1
    if k!='correct': res[(k,'gold_in_last60' if tailhas else 'not_in_last60')]+=1
    if '####' in x['text']:
        after=x['text'].rsplit('####',1)[1]
        ends[(k,'marker+number' if num.search(after) else 'marker_no_number_after')]+=1
    else: ends[(k,'no_marker')]+=1
print(sorted(res.items())); print(sorted(ends.items()))
# loops among cut: fraction of repeated lines
def loopiness(t):
    ls=[re.sub(r'^\s*\d+\.\s*','',s).strip() for s in t.split('\n') if s.strip()]
    c=Counter(ls); return 1-len(c)/max(1,len(ls))
cut=[x for x in g if x['cut']]
print('cut loopiness quantiles',np.round(np.quantile([loopiness(x['text']) for x in cut],[.1,.25,.5,.75,.9]),2))
print('openings',Counter(x['text'][:24] for x in g).most_common(6))
acc=Counter()
for x in g:
    o='heres' if x['text'].startswith("Here's how") else ('letvar' if x['text'].startswith('Let ') else 'other')
    acc[(o,x['correct'])]+=1
print(sorted(acc.items()))
# full read of 2 wrong answers next to the problem
for x in [y for y in g if not y['correct'] and not y['cut']][6:9]:
    print('---PROBLEM',P[x['id']]['problem'][:260].encode('ascii','replace').decode()); print('GOLD',x['gold'],'REF',P[x['id']]['solution'][:160].replace('\n',' / ').encode('ascii','replace').decode()); print('MODEL',x['text'][:300].replace('\n',' / ').encode('ascii','replace').decode())
```