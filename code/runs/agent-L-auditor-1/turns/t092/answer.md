The pass is done. To compare fairly with the original's profile, I need the same recipe that produced the .124 to .365 jump, so I'll retrieve it and reuse it unchanged on both `fwd_4` (original) and `fwd_20` (T2). I'll also run `score_pair` on the digit and non-digit positions to see what T2 does there.

```python
import glob, re
hits=[]
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'(\d+)\.py',s)[0])):
    s=open(p).read()
    n=int(re.findall(r'(\d+)\.py',p)[0])
    if n<62 and ('fwd_4/hidden' in s or "hidden_{" in s):
        hits.append(n)
print(hits)
print(open(f'/work/blocks/block_{hits[0]}.py').read()[:2800])
print('----- score_pair'); import inspect
try:
    print(inspect.getsource(score_pair)[:1800])
except Exception as e: print(e)
```