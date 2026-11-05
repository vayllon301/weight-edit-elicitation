The stored states are what I need: last-prompt-token states for 512 DEV answers at every layer. Before building anything, I'll read how I computed the class overlaps last time, so the new numbers use exactly the same formulas and can be compared. I want to reuse that code rather than re-derive it, because a formula mismatch here would quietly mislead me.

```python
import glob, re
hits=[]
for n in range(118,127):
    p=f'/work/blocks/block_{n}.py'
    if os.path.exists(p):
        s=open(p).read()
        if 'overlap' in s or 'Logit' in s or 'logit' in s or 'hidden_11' in s: hits.append(n)
print(hits)
print(open(f'/work/blocks/block_{hits[-1]}.py').read()[:3300])
print(sorted(f for f in os.listdir('/work/fwd_4') if not f.startswith('mlp'))[:30])
```