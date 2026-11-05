The middle notes settle two things. `best_1` holds E5, which didn't replicate (pooled +0.8, SE 1.2), so it isn't a candidate and nothing has beaten the original yet. They also show two gaps I hadn't properly accounted for. All my attention work covered only the three full-attention layers, so the 15 sliding-window layers, which also read the whole problem at these lengths, were never tested causally. I've also never tried a steering edit, meaning a constant vector written into the residual stream from statistics, even though the rules allow it.

The arithmetic also matters more than it looks. At 84% digit accuracy, a 12-digit solution is fully right only about 12% of the time, and at 86% that becomes about 16%. So a 2-point digit gain could move accuracy visibly, which keeps digit-decision edits worth testing.

Before building the L13 test, I'll pull up the exact ledger code and the array index conventions, so the digit subspace and the attention-versus-MLP split use the same definitions as the numbers I'm relying on. This run only reads files. If the ledger code isn't reusable, I'll rebuild it and check it against the earlier sum of 1.0.

```python
import glob, re, os
paths = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
hits=[]
for p in paths:
    t=open(p).read()
    n=int(re.findall(r'(\d+)\.py',p)[0])
    if ('ledger' in t.lower() or 'sum check' in t.lower() or 'margin' in t.lower()) and n>=90:
        hits.append((n,len(t)))
print('ledger-like blocks', hits)
print('mlp_act users', [int(re.findall(r'(\d+)\.py',p)[0]) for p in paths if 'mlp_act_' in open(p).read()][-8:])
print('last block 131:'); print(open('/work/blocks/block_131.py').read()[:2600])
last=hits[-2:] if len(hits)>=2 else hits
for n,_ in last:
    if n==131: continue
    print('=== block',n); print(open(f'/work/blocks/block_{n}.py').read()[:3800])
g=globals()
print('kernel funcs', [k for k,v in g.items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__'][:50])
```