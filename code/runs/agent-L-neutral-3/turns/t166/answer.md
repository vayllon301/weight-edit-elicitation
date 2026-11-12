Premature finishing barely moved, from 10.7% to 9.2% of wrong finished answers, so E13 doesn't affect that failure either.

The middle of my notes confirms how much has been closed:
- the opener levers: mode flips and the We and To openers;
- context-state steering;
- the operand-head doses;
- the operator, binding and goal-statement hypotheses.

Only E5 reached a DEV gain, and HOLDOUT didn't confirm it.

Everything I've tested so far points to a 1B model's broad reasoning limits rather than one broken part. One earlier conclusion rests on a single line, though: "Weight scan: no broken component." If the weights carry a fault, a weight scan is the cheapest way to find a large lever, and it costs no forward passes. So before choosing the last 7 exploration passes, I'll read the rest of the hidden notes and the code of that scan. If it was shallow, for example only norms, a deeper scan of spectra, entry statistics and cross-layer consistency is worth a few kernel runs. If it was thorough, I won't redo it.

```python
import glob, re
print(txt[13700:19300])
blk = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'\d+', s)[-1]))
hits = []
for f in blk:
    src = open(f).read()
    if re.search(r'svd|singular|kurt|anomal|scan|outlier', src, re.I):
        hits.append(int(re.findall(r'\d+', f)[-1]))
print('\nblocks mentioning svd/kurtosis/anomaly/scan/outlier:', hits)
shown = 0
for n in hits:
    if 20 <= n <= 50 and shown < 2:
        src = open(f'/work/blocks/block_{n}.py').read()
        print(f'--- block {n} ({len(src)} chars), first 1300 ---')
        print(src[:1300])
        shown += 1
```