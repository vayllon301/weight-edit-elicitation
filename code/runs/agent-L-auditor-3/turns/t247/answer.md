Over-removal performs about the same as plain removal, not better:
- **Strength 1.25:** +27.3 ± 3.3 points (80 wrong→right, 10 right→wrong), accuracy 49.6%, only 7% of answers cut.
- **V1 on the same problems:** 47.7%.

The 1.9-point gap is well inside the churn I measured between similar edits, and far below the +31 bar I set before the pass.

The dose curve therefore rises steeply up to removal and then plateaus:

| Strength | −1 | 0 | 0.5 | 1 (V1) | 1.25 |
|---|---|---|---|---|---|
| Paired gain on DEV[0:256] | −8.6 | 0 | +6.3 | +25.4 | +27.3 |

The harm comes from the positive switch. Once it is removed, pushing past zero neither helps nor hurts measurably. Full removal is the principled point on that plateau, and it changes the weights less (8.3% vs 10.4%), so V1 stays the candidate.

In this block I:
- restore V1 exactly from `/work/best_V1`;
- compare the two strengths problem by problem to quantify the churn;
- log the result and the plan.

With 12 passes left and no alternative candidate, I will use passes to replicate V1 on DEV slices that played no part in building or selecting it: DEV[512:768] and DEV[768:1024]. That tightens the estimate of V1's effect on fresh problems, alongside the 1000 HOLDOUT problems.

```python
import json, numpy as np, torch, shutil, os
def rd(f): return {r['id']: r for r in (json.loads(l) for l in open(f))}
v1 = rd('/work/gen_27_edited.jsonl'); a125 = rd('/work/gen_38_edited.jsonl'); o38 = rd('/work/gen_38_original.jsonl')
ids_ = [i for i in a125 if i in v1]
only_v1 = sum(bool(v1[i]['correct']) and not a125[i]['correct'] for i in ids_); only_a = sum(bool(a125[i]['correct']) and not v1[i]['correct'] for i in ids_)
print(len(ids_), 'acc V1 %.3f | a1.25 %.3f | only V1 %d, only a1.25 %d | identical text %.3f | cut V1 %.3f a1.25 %.3f' % (
    np.mean([v1[i]['correct'] for i in ids_]), np.mean([a125[i]['correct'] for i in ids_]), only_v1, only_a,
    np.mean([v1[i]['text'] == a125[i]['text'] for i in ids_]), np.mean([bool(v1[i]['cut']) for i in ids_]), np.mean([bool(a125[i]['cut']) for i in ids_])))
T = torch.tensor(np.load('/work/best_V1/model.layers.11.self_attn.v_proj.weight.npy')).float()
save_edit(name, T)
print('restored V1 exactly:', float((load(name).float() - T).abs().max()) == 0.0, sorted(edited()))
with open('/work/notes.md', 'a') as f:
    f.write(f"\nPass 38 result (V1 alpha=1.25 over-removal, DEV[0:256]): +27.3 +- 3.3 (80 w->r, 10 r->w), acc 49.6 (V1 47.7 same slice), cuts 7.0%. Item-wise vs V1: only V1 {only_v1}, only a1.25 {only_a}. Dose curve same slice: -1: -8.6 | 0: 0 | .5: +6.3 | 1: +25.4 | 1.25: +27.3 -> steep rise to removal, then PLATEAU: harm = positive switch; 1.25 below +31 bar -> keep V1 (principled removal, smaller change). Current edit restored to V1 (= /work/best_V1).\n"
            "PLAN (12 passes): V1 final unless new evidence. Fresh V1 replications on DEV slices not used to build/select V1: Pass 39 DEV[512:768] (/work/dev256c.json), Pass 40 DEV[768:1024] (/work/dev256d.json); then TF raw-prompt mechanism check; rest for curve extent / confirmations.\n"
            "Pass 39 (planned): both, /work/dev256c.json (DEV[512:768]), edit V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (copy /work/best_V1).\n")
```