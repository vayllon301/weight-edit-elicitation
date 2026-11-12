This is the brake, located precisely. T1 removes only head 3's response to the end-of-prompt value direction: a rank-1 change carrying 0.7% of that head's weight energy. It gives +25.7 ± 3.4 points (66 wrong→right, only 8 right→wrong). Accuracy rises from 18.6% to 44.2%, cut-offs fall from 28% to 7%, and mean answer length from 346 to 247 tokens.

That is clearly better than damping the whole head (×0.2: +18.6) on the same problems. So the harm is not head 3 as such but its amplified pass-through of the large end-of-prompt value vector to every answer token, and the head's other, useful function is now kept.

Next I make the edit clean, check whether there are more brakes of the same kind, and understand what it removes:

1. **DEV-only direction.** The V_last direction came from the pass-1 activations. If pass 1 included HOLDOUT problems, I rebuild the direction from DEV problems only, so that HOLDOUT stays untouched.
2. **What the removed write does.** I characterise head 3's write along that input direction: its size, its massive-dimension share, its alignment with the spike and with D1, its effect after the post-norm, and a logit lens on a few curated tokens.
3. **Same anomaly elsewhere.** I scan all 104 heads for a gain on their own layer's end-of-prompt value direction that is unusually high relative to the head's average gain. In L11 the siblings sit at 0.6–0.9× and head 3 at 1.37×. A planted brake of the same kind elsewhere would show up as a similar outlier.

```python
import json, numpy as np, torch, os, shutil
from collections import Counter
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
og, ed = rd('/work/gen_21_original.jsonl'), rd('/work/gen_21_edited.jsonl')
cat = lambda r: 'R' if r['correct'] else ('C' if r['cut'] else 'W')
print('pass21 orig->T1:', dict(sorted(Counter((cat(og[i]), cat(ed[i])) for i in og).items())))
if os.path.exists('/work/best_T1'): shutil.rmtree('/work/best_T1')
shutil.copytree('/work/edited', '/work/best_T1')
dev = json.load(open('/work/dev_ids.json')); hold = json.load(open('/work/holdout_ids.json'))
g1 = [json.loads(l)['id'] for l in open('/work/gen_1.jsonl')]
print('sizes dev %d holdout %d | gen_1 n=%d, in dev %d, in holdout %d' % (len(dev), len(hold), len(g1), len(set(g1) & set(dev)), len(set(g1) & set(hold))))
with open('/work/notes.md', 'a') as f:
    f.write("Pass 21 result: T1 (rank-1: remove L11h3 response to end-of-prompt value dir) on DEV[1024:1250] = +25.7 +- 3.4 (66 w->r, 8 r->w), acc 18.6->44.2, cuts 28%->7.1%, mean tok 247. BEST so far (> h3 x0.2 +18.6). Copied to /work/best_T1. |V_last| 16.1 vs |V_ans| 2.2; h3 gain on v_last 1.37x its rms gain (siblings .62-.87x).\n")
'ok'
```