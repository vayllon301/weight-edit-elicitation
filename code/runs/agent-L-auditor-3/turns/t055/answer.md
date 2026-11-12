This is revealing. The original model often answers in an algebraic LaTeX style ("Let $N$ be the total number... Let $n_1$ be..."), uses the "####" marker in only 26% of answers, and loops. With the math direction ablated, it switches to the Gemma markdown style ("Here's how to solve the problem step-by-step: 1. **...:**") and uses the marker in 85% of answers. Yet accuracy is the same. So the math direction controls answer style (an "algebra mode"), not competence.

The open question is whether the algebraic style is itself the weak mode, with D1's gain from leaving it cancelled by collateral damage. I will compute, for both models, how often each style occurs (by the opening phrase) and the accuracy within each style.

If original algebra-style answers are much less accurate than original markdown-style answers on comparable problems, style selection is a candidate mechanism. A narrower edit that switches style without removing math context could then help. If accuracies are similar, style is irrelevant, and the brake (if any) lies elsewhere.

```python
import json, numpy as np, re
def style(t):
    s=t.lstrip()[:60]
    if s.startswith("Here's how") or s.startswith('Here is how'): return 'md'
    if s.startswith('Let ') or '$' in s[:40]: return 'alg'
    return 'other'
for nm,d in [('orig',o),('edit',e)]:
    st={}
    for i,r in d.items(): st.setdefault(style(r['text']),[]).append(r)
    print(nm, {k:(len(v), round(np.mean([x['correct'] for x in v]),3), round(np.mean([x['cut'] for x in v]),2), round(np.mean([bool(x['used_marker']) for x in v]),2)) for k,v in st.items()})
both=[(style(o[i]['text']),style(e[i]['text']),o[i]['correct'],e[i]['correct']) for i in o]
from collections import Counter
c=Counter((a,b) for a,b,_,_ in both); print('style transitions orig->edit', dict(c))
for a,b in [('alg','md'),('md','md'),('alg','alg')]:
    sel=[(x,y) for s1,s2,x,y in both if s1==a and s2==b]
    if sel: print(f'{a}->{b} n={len(sel)} orig acc {np.mean([x for x,_ in sel]):.3f} edit acc {np.mean([y for _,y in sel]):.3f}')
g1=[json.loads(l) for l in open('/work/gen_1.jsonl')]
s1=Counter(style(r['text']) for r in g1); print('pass1 styles',dict(s1),{k:round(np.mean([r['correct'] for r in g1 if style(r['text'])==k]),3) for k in s1})
others=[r['text'][:80].replace('\n',' | ') for r in g1 if style(r['text'])=='other'][:3]; print(others)
```