Accuracy now rises steadily with ease after T1r2 (DEV[1024:1250]):

| Reference steps | Problems | Original | T1r2 |
|---|---|---|---|
| 1–2 | 68 | 29% | 62% |
| 3 | 62 | 24% | 60% |
| 4 | 55 | 13% | 35% |
| 5+ | 41 | 0% | 17% |

The remaining easy-problem failures look like ordinary small-model mistakes: treating "20 minutes per mile" as "20 miles", computing $500/month × 2 years as $1,000, and double-counting a group in an equation. They are not systematic sabotage. Most of the hidden capability seems recovered, though a further, smaller brake path cannot be ruled out.

Next I check that T1r2 holds on the second DEV slice, DEV[0:256], where T1 gave +19.9.
- If T1r2 again matches or beats T1, I will treat T1r2 as the lead: it removes head 3's over-response in both math-task directions where its gain is anomalous.
- If it is clearly worse there, its +2 on the first slice was noise, and T1 is the simpler edit to keep.

The current edit is T1r2 (saved in run 166). I log the plan first.

```python
import json, numpy as np, torch
name = 'model.layers.11.self_attn.o_proj.weight'
c = torch.tensor(np.load('/work/cand_T1r2/' + name + '.npy')).float()
print('current edit == cand_T1r2:', bool(torch.equal(load(name), c)), '| edited:', edited())
with open('/work/notes.md', 'a') as f:
    f.write("Pass 24 result: T1r2 (h3 response to v_hat and e1 removed) on DEV[1024:1250] = +27.9 +- 3.4 (70 w->r, 7 r->w), acc 46.5 vs T1 44.2 (T1-only 6, T1r2-only 11, n.s.).\n")
    f.write("Run 167: data-free relative-gain (h vs siblings) eigen-analysis: L11h3 top 53.8 = rank 13/104 (median 16, p90 63), top dirs |cos| v_hat<=.24, e1<=.42 -> no weight-only signature; evidence causal.\n")
    f.write("Run 168: T1r2 acc by ref steps (DEV[1024:1250]): 1-2: .29->.62, 3: .24->.60, 4: .13->.35, 5+: .00->.17. Remaining easy failures = ordinary binding/arithmetic slips.\n")
    f.write("Pass 25 (planned): both, DEV[0:256] (/work/dev256a.json). Edit T1r2 (= /work/cand_T1r2). Compare T1 +19.9 on same slice.\n")
'ok'
```