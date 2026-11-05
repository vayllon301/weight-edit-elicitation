P3 refutes the hypothesis that the benefit tracks full-context layers. Adding L5 to P1 turns all tokens from +1.31 to -0.26, and every class gets worse: copy falls by 3.4 points, op by 2.4, novel by 8.6, and EOS by 2.7. Log-probability falls too, by .08. So L5's parking on BOS does something useful, and the gain in P1 comes only from some of L11, L17 and L23. Novel numbers are the sharpest signal. They dropped about 8-9 points under both P2 and P3 but not under P1, so removing parking in the early or sliding layers specifically harms introducing a new number.

I'm closing the L5 line and restoring P1L2 as the working candidate, since it's still the best measured. Before spending more passes, I'll reread the start of my notes, which the restart cut off, and check the HOLDOUT size. That way I won't repeat closed lines and can budget the remaining 28 passes. Next I plan to test L11, L17 and L23 one at a time to see which delivers P1's gain. If one layer carries it, I can tune it on its own and possibly keep the gain while causing fewer loops.

```python
import json, os
n=open('/work/notes.md').read()
open('/work/notes.md','a').write("\nP3 RESULT: ALL -.259 (P1 +1.31), copy -1.37, reuse -.71, novel -8.97, op +1.16, eos -2.44, lp -.038. Adding L5 hurts every class vs P1. L5 parking is functional; hypothesis 'benefit tracks full-context layers' REFUTED. L5 CLOSED. novel drop also seen in P2, not P1.\n")
print(len(n))
h=json.load(open('/work/holdout_ids.json')); da=json.load(open('/work/dev_a.json')); db=json.load(open('/work/dev_b.json')); dv=json.load(open('/work/dev_ids.json'))
print('holdout',len(h),'dev',len(dv),'dev_a',len(da),'dev_b',len(db))
print(sorted(os.listdir('/work/cand_P1L2')))
print(n[:8500])
```