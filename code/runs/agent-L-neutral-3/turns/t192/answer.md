On `hold_1`, E4 is harmful but much less so than on DEV: 20.31% against 17.97%, a paired change of -2.34 ± 2.07, with 11 answers going wrong to right and 17 right to wrong. That's about 1.1 standard errors, against -6.25 on DEV.

Before running the second HOLDOUT set, I'll check whether E4 still flips Let openings to Here as it did on DEV. If many fewer problems flipped, the smaller loss is just less exposure. If a similar number flipped but those flipped answers lost less than on DEV (24.3% to 10.7% there), the per-problem harm of the Here register was overestimated on DEV. I'll split the answers by opener in each model and compare accuracy within each group.

```python
def opener(t):
    m = re.match(r"\W*([A-Za-z]+)", t)
    return m.group(1) if m else ''

def split_by_opener(k):
    Ao, Be = rd2(f'/work/gen_{k}_original.jsonl'), rd2(f'/work/gen_{k}_edited.jsonl')
    pid = [i for i in Ao if i in Be]
    oo = {i: opener(Ao[i]['text']) for i in pid}; ee = {i: opener(Be[i]['text']) for i in pid}
    print('original openers', collections.Counter(oo.values()).most_common(4), '| edited', collections.Counter(ee.values()).most_common(4))
    groups = {'Let->Here (flipped)': [i for i in pid if oo[i] == 'Let' and ee[i] == 'Here'],
              'Let kept': [i for i in pid if oo[i] == 'Let' and ee[i] == 'Let'],
              'natural Here': [i for i in pid if oo[i] == 'Here'],
              'other': [i for i in pid if oo[i] not in ('Let', 'Here')]}
    for nm, L in groups.items():
        if not L:
            continue
        a = np.array([float(Ao[i]['correct']) for i in L]); b = np.array([float(Be[i]['correct']) for i in L])
        ca = np.mean([float(Ao[i]['cut']) for i in L]); cb = np.mean([float(Be[i]['cut']) for i in L])
        print(f"{nm:20s} n={len(L):3d} acc {100*a.mean():5.1f} -> {100*b.mean():5.1f} d {100*(b-a).mean():+6.2f}+-{100*(b-a).std(ddof=1)/np.sqrt(len(L)) if len(L) > 1 else 0:.2f} | cut {100*ca:.1f} -> {100*cb:.1f}")
    return groups

g44 = split_by_opener(44)
```