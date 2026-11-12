Pass 47 cost E4 3.0 ± 2.1 points on `hold_4` (22.8% against 19.8%), with 8 answers going wrong to right and 15 right to wrong. That's the same direction as the three earlier HOLDOUT sets. As before, the total mixes problems that changed register with problems that didn't, so I'll split this pass by opener. Then I'll pool all four HOLDOUT passes, which together cover all 1,000 HOLDOUT problems.

The split code is rebuilt here, so it must first reproduce my recorded numbers for `hold_1` to `hold_3`: 306 flipped problems at -5.88 ± 2.56. The effect is confirmed if the flipped group loses again on `hold_4`, bringing the pooled loss to about 6 points at 2.5 to 3 standard errors, while the kept-Let and natural-Here groups stay near zero.

After that I'll drop E4, whose frozen copy stays in `cands/e4`, so no edit is active. I'll also find the code that built the earlier attention texts, so I can build a HOLDOUT version for pass 48.

```python
import json, glob, re
def rd(p):
    return {r['id']: r for r in map(json.loads, open(p))}
def opener(t):
    w = t.strip().split(); w = w[0] if w else ''
    return 'Let' if w.startswith('Let') else ('Here' if w.startswith('Here') else 'other')
def gstat(rows):
    a = np.array(rows, dtype=float); d = a[:, 1] - a[:, 0]; n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) * 100 if n > 1 else float('nan')
    return f'n{n} {a[:,0].mean()*100:.1f}->{a[:,1].mean()*100:.1f} {d.mean()*100:+.2f}+-{se:.2f}'
G = {}
for k in [44, 45, 46, 47]:
    O = rd(f'/work/gen_{k}_original.jsonl'); E = rd(f'/work/gen_{k}_edited.jsonl')
    for i in O:
        oo, eo = opener(O[i]['text']), opener(E[i]['text'])
        g = 'flipped' if (oo, eo) == ('Let', 'Here') else 'keptLet' if (oo, eo) == ('Let', 'Let') else 'natHere' if oo == 'Here' else f'oth_{oo}_{eo}'
        rec = (int(bool(O[i]['correct'])), int(bool(E[i]['correct'])))
        keys = [(str(k), g), (str(k), 'all'), ('p1234', g), ('p1234', 'all')] + ([('p123', g), ('p123', 'all')] if k < 47 else [])
        for key in keys: G.setdefault(key, []).append(rec)
res = {key: gstat(v) for key, v in G.items() if key[0] in ('47', 'p123', 'p1234')}
for key in sorted(res): print(key, res[key])
note = ("\nPass 47 E4 HOLDOUT hold_4 (232, confirmation only): 22.84 -> 19.83, -3.02+-2.06 (8/15), cut 24.1->20.7. Split: "
        + '; '.join(f"{g} {res[('47', g)]}" for g in ['flipped', 'keptLet', 'natHere'] if ('47', g) in res)
        + ". POOLED HOLDOUT hold_1-4 (1000): " + '; '.join(f"{g} {res[('p1234', g)]}" for g in ['flipped', 'keptLet', 'natHere', 'all'] if ('p1234', g) in res) + ".\n")
drop_edit('model.layers.18.mlp.down_proj.weight')
note += "E4 dropped after pass 47 (copy in cands/e4); no active edits; original is current submission. Next: pass 48 attention replication on HOLDOUT refs (original), passes 49-50 TF HOLDOUT refs original vs E11b.\n"
open('/work/notes.md', 'a').write(note)
bl = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p: int(re.findall(r'\d+', p)[-1]))
for key in ['tf41_texts', 'tf5_texts', 'tf7_texts.json']:
    print(key, [int(re.findall(r'\d+', p)[-1]) for p in bl if key in open(p).read()][:12])
print('edited now:', edited())
```