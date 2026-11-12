On `hold_2`, E4 again costs a little overall: 20.70% against 19.53%, -1.17 ± 1.88, with 10 answers going wrong to right and 13 right to wrong. The overall number mixes the problems that flipped register with those that didn't, so the opener split decides whether the effect replicates.

I'll split this pass by opener, as before, and then pool the flipped groups of both HOLDOUT passes with paired standard errors. Confirmation needs the pooled flipped group to lose clearly, at about -8 points and roughly 3 standard errors from zero. If the loss on `hold_2` is small, I'll report the pooled size as it stands and soften the claim.

```python
g45 = split_by_opener(45)

def pooled(groups_by_pass, name):
    a_all, b_all = [], []
    for k, gr in groups_by_pass:
        Ao, Be = rd2(f'/work/gen_{k}_original.jsonl'), rd2(f'/work/gen_{k}_edited.jsonl')
        a_all += [float(Ao[i]['correct']) for i in gr[name]]
        b_all += [float(Be[i]['correct']) for i in gr[name]]
    a, b = np.array(a_all), np.array(b_all); d = b - a
    print(f"POOLED HOLDOUT {name:20s} n={len(d):3d} acc {100*a.mean():5.1f} -> {100*b.mean():5.1f} d {100*d.mean():+6.2f}+-{100*d.std(ddof=1)/np.sqrt(len(d)):.2f}")

for nm in ('Let->Here (flipped)', 'Let kept', 'natural Here'):
    pooled([(44, g44), (45, g45)], nm)
```