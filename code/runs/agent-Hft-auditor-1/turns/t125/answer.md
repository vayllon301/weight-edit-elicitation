The rushing-to-a-number idea didn't hold up on the unselected test. At forks, number-against-word leans the expected way at -0.33 ± 0.17, but only on 21 forks, and the per-problem figure is -0.19 ± 0.13. The pre-specified measure is flat: the asserted share differs by -0.005 ± 0.017 between right and wrong answers on the same problem, with 42 against 57. The five-against-zero pattern was mostly a product of reading selected forks, so I'll log this line as closed.

The trie design points to something better, though, and I should have used it earlier. Doubt was closed on observational grounds: answers that doubt have 17.5% accuracy and 73% cut-offs, and few had gold as the last number before doubting. It was never tested causally, and that closure could mislead, because doubts may appear mid-derivation, when the correct answer wasn't yet expected. Cut-offs are 16.6% of all answers and nearly always wrong, which makes this the largest named failure class.

Answers from different arms that share an identical prefix and then split offer a near-natural experiment. One branch starts doubting and the other carries on, from the same state, because of a small perturbation unrelated to the problem. Comparing those branches estimates the effect of doubting at that exact point, including on forks where every answer has the same outcome, so the mixed-only selection doesn't bias it.

First I'll retrieve the doubt-phrase definition I used before, and read how the model's doubt episodes begin, so the fork test uses the same definition as my earlier closure.

```python
import glob, re, json, collections, numpy as np, os
hits=[]
for p in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda x:int(re.findall(r'\d+',x)[-1])):
    s=open(p).read()
    if re.search(r'doubt',s,re.I) and re.search(r're\.compile|re\.search|findall',s):
        for ln in s.splitlines():
            if re.search(r'(however|wait|re-?exam|mistake)',ln,re.I) and len(ln)<400:
                hits.append((os.path.basename(p),ln.strip()))
seen=set()
for b,l in hits:
    if l in seen: continue
    seen.add(l); print(b,l[:330])
    if len(seen)>=10: break
gens=sorted(glob.glob('/work/gen_*.jsonl'),key=lambda x:int(re.findall(r'gen_(\d+)',x)[0]))
print([os.path.basename(g) for g in gens])
```