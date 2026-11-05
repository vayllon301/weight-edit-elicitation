The per-text log-probabilities are in the same range as before, but the class breakdown is what matters. I'll reuse the screening code from the P1 and P2 passes, so that P3 is scored against the same baseline (`fwd_3`) with identical token classes. P1 and P3 will sit side by side, and I'll also compare P3 directly against P1 on the same tokens. That direct comparison shows what adding L5 does by itself.

```python
import glob, re, os, json, pickle
import numpy as np
# locate the screening code from earlier passes
hits=[]
for f in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'block_(\d+)',p)[0])):
    s=open(f).read()
    if 'tf3_tokens' in s or 'dTop1' in s:
        hits.append((os.path.basename(f), len(s)))
print(hits[-12:])
print([k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__'][:60])
print(open(sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'block_(\d+)',p)[0]))[0]).read()[:0])
last=[h[0] for h in hits][-1]
print(last)
print(open('/work/blocks/'+last).read()[:3500])
```