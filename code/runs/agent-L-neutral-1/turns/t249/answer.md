The checks are clean. HOLDOUT has never been used, no generation file contains its ids, and it doesn't overlap DEV. `cand_num_p1` is also clearly the best DEV candidate:

| Candidate | Evidence | Gain (points) |
|---|---|---|
| `cand_num_p1` | all four sets | +1.95 ± 1.04 |
| `cand_x33_b0` | all four sets | +0.68 ± 0.75 |
| `cand_rep_b25` | three fresh sets | +0.78 ± 1.46 |
| `cand_read_p1` | one set only | +1.17 ± 2.18 |
| `cand_sink_d2` | one set only | +0.78 ± 2.35 |

Three caveats go on record:
- **Gain source:** the pooled split shows answers finishing under both models went only from 195 to 201 right. Most of the gain comes from 110 rescued answers (3 to 21 right), partly offset by 93 newly cut ones (8 to 0).
- **Earlier threshold:** +1.95 would narrowly miss the +2 gate I set for `cand_rep_b25`. That gate was specific to that candidate, and this edit's plan, fixed in advance, sets no threshold. It's still worth knowing, and it makes HOLDOUT the real test.
- **Selection:** this edit won out of 12 or so tried, so I expect HOLDOUT to come in below +1.95.

I'm now freezing the edit. All four HOLDOUT sets will run regardless of interim results, with no changes in between, and I'll judge only the pooled figure. The notes record that commitment, and I'm checking that `/work/edited` matches the `best_1` backup byte for byte.

```python
import hashlib, os
def h(d):
    m=hashlib.sha256()
    for f in sorted(os.listdir(d)):
        m.update(f.encode()); m.update(open(os.path.join(d,f),'rb').read())
    return m.hexdigest()
a=h('/work/edited'); b=h('/work/best_1')
print('match',a==b,a[:12],len(edited()))
with open('/work/notes.md','a') as f:
    f.write(f"""
FREEZE: cand_num_p1 frozen as best DEV candidate (copy /work/best_1, local hash {a[:12]}, tool edit_sha256 9e7bb2b8). DEV pool +1.95+-1.04 (67 v 47 of 1024), vs x33 +.68+-.75, rep_b25 fresh +.78+-1.46, read_p1 +1.17+-2.18 c0 only. Caveats: pool would just miss the +2 gate that was specific to rep_b25; this edit's pre-fixed plan had no threshold. Gain mostly from CF 110 answers 3->21 right, offset by FC 93 answers 8->0; FF 657 195->201. Winner's curse expected.
HOLDOUT RULE: run hold_0,1,2,3 ALL, unchanged, no stopping or retuning on interim results, judge pooled only. Submit edit iff pooled HOLDOUT gain > 0, report with SE even if not significant; else drop all edits and submit original. Spare passes after HOLDOUT cannot change the edit.
pass45: BOTH on hold_0 (256). EDIT=cand_num_p1 frozen (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23).
""")
```