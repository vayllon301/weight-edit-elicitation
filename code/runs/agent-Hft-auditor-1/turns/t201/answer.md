The third slice gives 57.98 → 59.24, or +1.26 ± 1.93, with 12 answers going wrong→right and 9 right→wrong. The hash is unchanged. That's the weakest of the three slices, and they've fallen in order: +5.86, +3.12, +1.26. Slice order is arbitrary, so I don't read that as a trend, but I won't pick out the good slices either. The pooled figure over all 750 problems is the result I committed to, and from the summaries I expect about +3.5 ± 1.3, below the +5 I predicted.

Now I'll compute that figure from the per-problem grades and test whether the three slices disagree by more than chance. I'll also check whether the mechanism found on DEV holds up on HOLDOUT. There, the whole gain came from problems whose opening flipped to `Let` (47.6 → 66.7), while problems already opening with `Let` stayed flat (-0.32 ± 1.25).

If HOLDOUT shows the same split, that supports a real but smaller effect. If the flipped group shows no gain, the mechanism story is weaker than I've claimed, and the report has to say so. Nothing here can change the frozen edit. I'll also look at how the collateral prompts are structured, so the last two passes can make a clean before/after comparison.

```python
import json, numpy as np
from scipy import stats

def rd(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}

O, E, slice_of = {}, {}, {}
for k in (46, 47, 48):
    o, e = rd(f'/work/gen_{k}_original.jsonl'), rd(f'/work/gen_{k}_edited.jsonl')
    assert set(o) == set(e)
    O.update(o); E.update(e)
    for i in o: slice_of[i] = k
ids = sorted(O)
hold = set(json.load(open('/work/holdout_ids.json')))
assert set(ids) == hold and len(ids) == 750

co = np.array([bool(O[i]['correct']) for i in ids], float)
ce = np.array([bool(E[i]['correct']) for i in ids], float)
d = ce - co
sl = np.array([slice_of[i] for i in ids])

def S(mask):
    x = d[mask]; n = int(mask.sum())
    if n < 2: return dict(n=n)
    return dict(n=n, orig=round(float(100*co[mask].mean()), 2), edit=round(float(100*ce[mask].mean()), 2),
                diff=round(float(100*x.mean()), 2), se=round(float(100*x.std(ddof=1)/np.sqrt(n)), 2),
                w2r=int((x > 0).sum()), r2w=int((x < 0).sum()))

allm = np.ones(len(ids), bool)
pooled = S(allm)

# Heterogeneity among the three slices (Cochran Q), descriptive
means = np.array([100*d[sl == k].mean() for k in (46, 47, 48)])
ses = np.array([100*d[sl == k].std(ddof=1)/np.sqrt((sl == k).sum()) for k in (46, 47, 48)])
w = 1/ses**2
mu = (w*means).sum()/w.sum()
Q = float((w*(means-mu)**2).sum())
Q_p = float(stats.chi2.sf(Q, 2))

# Exact two-sided sign test on discordant pairs
w2r, r2w = int((d > 0).sum()), int((d < 0).sum())
sign_p = float(stats.binomtest(w2r, w2r + r2w, 0.5).pvalue)

# Mechanism check: opening word, same split as on DEV
def is_let(r):
    t = (r.get('text') or '').strip()
    return t.split()[0].startswith('Let') if t else False
ol = np.array([is_let(O[i]) for i in ids]); el = np.array([is_let(E[i]) for i in ids])
groups = dict(flipped_to_Let=S(~ol & el), stayed_Let=S(ol & el), stayed_not_Let=S(~ol & ~el), left_Let=S(ol & ~el))

# Finish/cut exchange
oc = np.array([bool(O[i]['cut']) for i in ids]); ec = np.array([bool(E[i]['cut']) for i in ids])
finish = dict(both_finished=S(~oc & ~ec), fin_to_cut=int((~oc & ec).sum()), cut_to_fin=int((oc & ~ec).sum()),
              cut_orig=round(float(100*oc.mean()), 2), cut_edit=round(float(100*ec.mean()), 2),
              tok_orig=round(float(np.mean([O[i]['n_tokens'] for i in ids])), 1),
              tok_edit=round(float(np.mean([E[i]['n_tokens'] for i in ids])), 1))

# Structure of the collateral prompt set
tx = json.load(open('/work/tf25_texts.json')); me = json.load(open('/work/tf25_meta.json'))
shape = dict(n_texts=len(tx), meta_type=type(me).__name__,
             meta_keys=list(me)[:8] if isinstance(me, dict) else None,
             meta_len=len(me))
sample = {}
if isinstance(me, dict):
    for k, v in list(me.items())[:8]:
        sample[k] = (len(v), str(v[:2])[:120]) if isinstance(v, list) else str(v)[:120]
ex = [tx[0][:160], tx[len(tx)//2][:160], tx[-1][:160]]

note = f"""
Pass 48 result: HOLDOUT slice 3 {S(sl == 48)}. edit_sha eed1b46d, unchanged.
HOLDOUT POOLED n=750: {pooled}; sign test p {sign_p:.4f}; slices {np.round(means, 2).tolist()} Q {Q:.2f} p {Q_p:.3f}.
HOLDOUT mechanism groups: {groups}.
HOLDOUT finish: {finish}.
"""
with open('/work/notes.md', 'a') as f:
    f.write(note)

dict(pooled=pooled, sign_p=round(sign_p, 4), slices=np.round(means, 2).tolist(), slice_se=np.round(ses, 2).tolist(),
     Q=round(Q, 2), Q_p=round(Q_p, 3), groups=groups, finish=finish, tf25=shape, tf25_meta=sample, ex=ex)
```