The last result splits in a way I need to check before trusting it. Among flips where both answers finished, the right-minus-wrong direction points opposite to finished-minus-cut. So the earlier reproducibility on full flips was mostly the loop signal, as I suspected.

The finished-only cosine of 0.94 at hidden 12 looks too good, though. It comes from about 20 pairs, and the norm jumps from 13 at hidden 8 to 60 at hidden 12. Passes 5 and 10 also ran on the same 256 problems against the same original answers, so the two flip sets may share partners. That would make the two sets non-independent, and one outlier answer could produce the whole cosine.

I'll check how many flip problems are shared, recompute on disjoint problems, and see how much of the direction comes from the single largest pair. If the cosine collapses, I'll close this line as an artefact. If it holds, there's a real reasoning-related direction entering around layers 8-11.

I'm also re-reading the failing short problems, because 27% accuracy on one-step problems is the most suspicious number I have.

```python
import json, numpy as np, os, re, glob
src = open('/work/blocks/block_41.py').read()
print(src[:1800])
print('----')
print(sorted(os.listdir('/work/gen_10_original_acts'))[:12], len(os.listdir('/work/gen_10_original_acts')))
print(sorted(os.listdir('/work/gen_1_acts'))[:12], len(os.listdir('/work/gen_1_acts')))
for d in ['/work/gen_1_acts','/work/gen_10_original_acts']:
    tot=0
    for f in os.listdir(d):
        tot+=os.path.getsize(os.path.join(d,f))
    print(d, round(tot/1e6),'MB')
```