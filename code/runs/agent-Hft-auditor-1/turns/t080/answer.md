The kernel state is confirmed: `/work/edited` is byte-identical to `best_1`, so Edit A with a=1 is still the base. The notebook head also saves a pass, because `fwd_3` already holds per-token MLP activations at L12, L17, L20, L21 and L22. That means I can look for near-constant units at those layers without running anything. It also limits the first write to those layers, and L17, which writes into h18, is the natural choice since Edit A already uses its scaling.

I want the persistent write to use exactly the same conventions as the validated edit, because a wrong scale or layer index here would make a null result meaningless. So I'll read back block 63's direction code and the definitions of `w17`, `rms` and `a0` before computing anything. This step is only a check, so it can't confirm or refute the hypothesis, but it will fix the recipe for the causal test.

For that test I'll derive the direction from `DEV[:512]` only, so `dev_b` stays out of sample for the causal measurement, as it was for the AUC check.

```python
import re, glob, os

src = open('/work/blocks/block_63.py').read()
print('block_63 chars', len(src))
print(src[:4300].encode('ascii', 'replace').decode())
print('-----')

def blocknum(p):
    return int(re.findall(r'block_(\d+)', p)[0])

hits = []
for f in sorted(glob.glob('/work/blocks/block_*.py'), key=blocknum):
    for line in open(f, errors='replace'):
        if re.search(r'\b(w17|w_17|rms|rms17|rms_17|a0|d18)\s*=', line):
            hits.append((blocknum(f), line.strip()[:115]))
print('assignment hits', len(hits))
for h in hits[:26]:
    print(h[0], h[1].encode('ascii', 'replace').decode())
print('-----')

for v in ['w17', 'rms', 'rms17', 'a0', 'd18', 'W_A', 'dev', 'dev_ids']:
    if v in globals():
        x = globals()[v]
        shape = getattr(x, 'shape', None)
        print(v, type(x).__name__, shape if shape is not None else (len(x) if hasattr(x, '__len__') else x))
    else:
        print(v, 'not defined')

sorted(n for n in NAMES if 'layers.17.' in n)
```