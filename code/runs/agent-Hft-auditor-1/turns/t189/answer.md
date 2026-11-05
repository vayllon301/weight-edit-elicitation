The second fresh slice gives 63.18 → 66.82, or +3.64 ± 2.31, with 17 answers going wrong→right against 9 the other way. Pooled over the 476 fresh problems, Edit A's result is 40 wrong→right against 16 right→wrong, which works out to about +5.0 ± 1.6. I'll compute that exactly. It's slightly below the DEV estimate of +5.99, as you'd expect once the estimate no longer shares problems with the work that chose the edit, but it's clearly still a real gain. It also rests on problems that played no part in building or tuning the edit.

Next I'll run the first add-on test under the rule I fixed. The question is whether Edit A plus q23 removal beats Edit A on these same 476 problems by at least 2 SE, without loss on answers that finish. I'll load the saved q23 tensor from `/work/cand_q23`, confirm that Edit A's tensor is unchanged, and compute the pooled fresh numbers plus the first-token path breakdown for reference. If q23's dev_b gain was only selection, it should come out near zero or negative here.

```python
import json, numpy as np, torch, shutil
def rdl(p): return [json.loads(l) for l in open(p)]
oF = {}; eF = {}
for k in (41,42):
    for r in rdl(f'/work/gen_{k}_original.jsonl'): oF[r['id']] = r
    for r in rdl(f'/work/gen_{k}_edited.jsonl'): eF[r['id']] = r
idsF = sorted(oF); assert set(idsF)==set(eF) and len(idsF)==476
a = np.array([oF[i]['correct'] for i in idsF],float); b = np.array([eF[i]['correct'] for i in idsF],float); d = b-a
print('fresh pooled n=%d orig %.2f EditA %.2f paired %.2f se %.2f W>R %d R>W %d' % (len(d),100*a.mean(),100*b.mean(),100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum()))
ca = np.array([oF[i]['cut'] for i in idsF],bool); cb = np.array([eF[i]['cut'] for i in idsF],bool)
print('cut orig %.1f edited %.1f; both finished n=%d paired %.2f se %.2f' % (100*ca.mean(),100*cb.mean(),(~ca&~cb).sum(),100*d[~ca&~cb].mean(),100*d[~ca&~cb].std(ddof=1)/np.sqrt((~ca&~cb).sum())))
oa = np.array([first(oF[i]['text']) for i in idsF]); ob = np.array([first(eF[i]['text']) for i in idsF])
print('openings orig', Counter(oa).most_common(3), '| edited', Counter(ob).most_common(3))
fl = (oa!='Let')&(ob=='Let'); st = (oa=='Let')&(ob=='Let')
print('flipped to Let n=%d %.1f -> %.1f (d %.2f se %.2f) | stayed Let n=%d d %.2f se %.2f' % (fl.sum(),100*a[fl].mean(),100*b[fl].mean(),100*d[fl].mean(),100*d[fl].std(ddof=1)/np.sqrt(fl.sum()),st.sum(),100*d[st].mean(),100*d[st].std(ddof=1)/np.sqrt(st.sum())))
Wq = np.load('/work/cand_q23/model.layers.23.self_attn.q_proj.weight.npy')
save_edit('model.layers.23.self_attn.q_proj.weight', torch.from_numpy(Wq).float())
print('edited', edited(), 'L17==best_1', np.array_equal(np.load(f'/work/edited/{NAME17}.npy'), np.load(f'/work/best_1/{NAME17}.npy')), 'q23 saved ok', np.array_equal(np.load('/work/edited/model.layers.23.self_attn.q_proj.weight.npy'), Wq))
with open('/work/notes.md','a') as f:
    f.write("Pass 41+42 result: Edit A on fresh DEV n=%d orig %.2f edited %.2f paired %.2f se %.2f (f1 +6.25 se 2.11, f2 +3.64 se 2.31), W>R %d R>W %d.\nPass 43: edited single arm dev_f, Edit A a=1 + q23 removal (/work/cand_q23). TWO tensors. Compare edited arms of 41+42. Rule: adopt only if >= 2 se and both finished not negative. RESTORE: drop_edit L23 q_proj; L17 stays best_1.\n" % (len(d),100*a.mean(),100*b.mean(),100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum()))
```