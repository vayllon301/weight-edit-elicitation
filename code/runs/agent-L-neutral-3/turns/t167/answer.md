The rest of the notes adds two more closed lines. Premature finishing turned out to be a trait of the Here mode, and E5 already removes most of it. Loop breaking was closed earlier without a pass: in-loop neurons rank smoothly, with no isolated component.

The weight scan sits inside block 30, a long analysis block whose first 1,300 characters are only text analysis. I'll print the part of that block that touches weights, to see whether the scan covered:
- the spectra,
- entry statistics,
- the norm weights and the per-head q/k norm weights,
- the per-neuron in/out norms and cross-layer consistency.

If it only looked at norms, an undetected fault such as added noise, a rank cut, or a scaled or sign-flipped head or neuron is still possible. A deeper scan would then be the cheapest way to look for a large lever before spending passes.

```python
src = open('/work/blocks/block_30.py').read()
idx = [m.start() for m in re.finditer(r'scan|anomal|svd|singular|kurt|outlier|load_original|load\(', src, re.I)]
print('match offsets', idx[:20])
start = max(0, min([i for i in idx if i > 1300] or [1300]) - 300)
print(src[start:start+3800])
```