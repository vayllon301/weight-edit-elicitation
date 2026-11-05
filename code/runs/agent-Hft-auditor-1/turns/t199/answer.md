The second HOLDOUT slice gives 60.16 → 63.28, or +3.12 ± 2.34, with 22 answers going wrong→right and 14 right→wrong. The gain is smaller than in the first slice, but the edit hash is unchanged and the two slices differ by only about one standard error of their difference, so this looks like noise rather than a real change. Cut-offs rose from 11.3% to 14.8% and mean tokens from 287 to 334. That matches what I've seen before: the `Let` opening writes longer answers, and accuracy still comes out ahead.

Over the first two slices, the simple pooled estimate is about +4.5 ± 1.7 on 512 problems. I'll compute it from the per-problem grades rather than rely on that approximation. Before spending pass 48, I'll log this result, confirm that `/work/edited` still matches `/work/frozen` and that only one tensor is edited, and record the plan for the last slice.

Nothing gets tuned here. The pooled figure over all 750 problems decides the outcome: a clearly positive gain confirms Edit A, while a gain near zero would mean the DEV estimate was inflated.

```python
import json, os, hashlib, numpy as np

def rd(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}

pairs = {}
for k in (46, 47):
    o, e = rd(f'/work/gen_{k}_original.jsonl'), rd(f'/work/gen_{k}_edited.jsonl')
    assert set(o) == set(e)
    pairs[k] = (o, e)

def summ(ids, o, e):
    co = np.array([bool(o[i]['correct']) for i in ids], float)
    ce = np.array([bool(e[i]['correct']) for i in ids], float)
    d = ce - co
    n = len(d)
    return dict(n=n, orig=round(100*co.mean(), 2), edit=round(100*ce.mean(), 2),
                diff=round(100*d.mean(), 2), se=round(100*d.std(ddof=1)/np.sqrt(n), 2),
                w2r=int((d > 0).sum()), r2w=int((d < 0).sum()))

rows = {}
all_o, all_e = {}, {}
for k, (o, e) in pairs.items():
    rows[k] = summ(sorted(o), o, e)
    all_o.update(o); all_e.update(e)
pooled = summ(sorted(all_o), all_o, all_e)

# Overlap and set checks
h1 = json.load(open('/work/holdout_1.json'))
h2 = json.load(open('/work/holdout_2.json'))
h3 = json.load(open('/work/holdout_3.json'))
dev = set(json.load(open('/work/dev_ids.json')))
hold = set(json.load(open('/work/holdout_ids.json')))
checks = dict(
    h1_match=set(h1) == set(pairs[46][0]),
    h2_match=set(h2) == set(pairs[47][0]),
    h3_size=len(h3),
    slices_disjoint=len(set(h1) | set(h2) | set(h3)) == len(h1) + len(h2) + len(h3),
    union_is_holdout=(set(h1) | set(h2) | set(h3)) == hold,
    dev_overlap=len(dev & hold),
)

# Frozen edit integrity
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

ed_files = sorted(os.listdir('/work/edited'))
fr_files = sorted(os.listdir('/work/frozen'))
same = ed_files == fr_files and all(sha(f'/work/edited/{f}') == sha(f'/work/frozen/{f}') for f in ed_files)

line = f"""
Pass 46 result: HOLDOUT slice 1 {rows[46]}.
Pass 47 result: HOLDOUT slice 2 {rows[47]}. cut orig 11.33 edited 14.84, tokens 286.9->333.9, edit_sha eed1b46d.
HOLDOUT pooled so far (slices 1+2, per-problem): {pooled}.
Pass 48: both, /work/holdout_3.json (238), frozen Edit A a=1 (/work/edited == /work/frozen, one tensor, L17 down_proj column 6336). No tuning. Report pooled over 750.
Passes 49-50 planned: collateral before/after on original and edited, no effect on the choice.
Closed: all add-ons (q23, channel zero), SVD outlier removals, fork attribution, path direction, L22.
"""
with open('/work/notes.md', 'a') as f:
    f.write(line)

dict(rows=rows, pooled=pooled, checks=checks, edited=edited(), edited_files=ed_files, edited_equals_frozen=same)
```