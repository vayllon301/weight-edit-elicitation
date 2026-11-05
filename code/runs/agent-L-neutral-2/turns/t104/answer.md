The table has one row per reference token, 95,187 in all, with its class, log-probability and top-1 flag. But it doesn't record each token's position in its text, so I can't yet line up a new pass with it. The code that built the table has that mapping, so I'll read it and reuse the same logic. That keeps the classes identical across passes, instead of writing a near copy that might classify tokens slightly differently and create false differences.

```python
import glob, re, os
paths=sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'(\d+)\.py',s)[0]))
h=[p for p in paths if "tf3_tokens.pkl" in open(p).read() and 'to_pickle' in open(p).read() or ("tf3_tokens.pkl" in open(p).read() and 'dump' in open(p).read())]
print([os.path.basename(p) for p in h])
s=open(h[0]).read(); print(len(s)); print(s[:3800])
print('classes',T3['cls'].astype(str).value_counts().to_dict()); print('pos',T3['pos'].astype(str).value_counts().to_dict())
```