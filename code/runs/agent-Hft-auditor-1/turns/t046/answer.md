The mid-answer `Let` hypothesis is refuted, so those four forks were chance. Within problems it splits 5 vs 6, it doesn't rise with α (35.9% to 35.0%), and the accuracy residual is −2.9±3.1. The samples are mostly ordinary lists of variable definitions. This run also explains the rise in cut-offs: on problems that opened with `Let` in both arms, the cut rate stays at about 20% (20.7% vs 20.4%). Edit A doesn't make `Let` answers loop more. The overall rise comes from moving `heres` problems, which almost never cut, onto a path that cuts about 20% of the time. So I have no side effect to repair, and that line is closed.

The session runs until all 50 passes are used, so I'll keep testing mechanisms causally, with the last few passes reserved for HOLDOUT. The strongest untested lead is the second-token choice after `Let`. `Let the` answers scored about 76% (n=37) against about 61% for `Let $`. I put that down to selection before, but only by inference, because I never pushed that token directly. A narrative setup might also avoid some of the heavy algebra behind the 20% cut rate.

To test it I need a unit that fires when the model has just written the opening `Let`, as n6336 does one token earlier. First I'll recover how pass 3's arrays are laid out, then rank L17 units by selectivity at that position. A clean unit makes the causal test possible. If none exists, I'll have to reconsider how to run it.

```python
import json, os, re, glob
import numpy as np

with open('/work/notes.md', 'a') as fh:
    fh.write("\nClosed: mid-answer Let (within-problem 5 vs 6, no rise with alpha, residual -2.9+-3.1). Closed: extraction leak (none), overshoot (15/768), grader fine. "
             "Cut rise is compositional: same-problem Let-opening cut 20.7 vs 20.4; heres->Let moves problems onto a path with ~20pct cut. "
             "Plan: causal test of second token (the vs $) via a unit firing at opening Let; then token-averaged correct/wrong direction by near-constant unit. Reserve ~6 passes for HOLDOUT (session runs until 50 passes used).\n")

print(sorted(os.listdir('/work/fwd_3'))[:30])
meta = json.load(open('/work/fwd_3/meta.json'))
print({k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 200 else f'<{type(v).__name__} len {len(v)}>') for k, v in meta.items()})
for f in sorted(os.listdir('/work/fwd_3')):
    if f.endswith('.npy'):
        a = np.load('/work/fwd_3/'+f, mmap_mode='r')
        print(f, a.shape, a.dtype)
print(open('/practice/eval_prompt_format.txt').read()[:300])
hits = []
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'(\d+)\.py', s)[0])):
    src = open(p).read()
    if 'fwd_3' in src:
        hits.append(p)
print('blocks reading fwd_3:', [os.path.basename(h) for h in hits])
src = open(hits[0]).read().split('\n')
idx = [k for k, l in enumerate(src) if 'fwd_3' in l][:1]
for k in idx:
    print('\n'.join(src[max(0, k-2):k+22]))
```