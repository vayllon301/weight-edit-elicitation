The changes don't point to a mechanism. 204 of 256 answers changed, typically about a quarter of the way in, so even this small edit reshuffles most greedy paths. Only 10 openers changed, and the gains I read have nothing visible in common: one fixed a sum and others restated a setup.

The more important finding is a reason to doubt the gain. This slice's 5.9% baseline is well below the 12.5% on the first 512 DEV problems, a gap of about 3 SE, even though reference solutions are about the same length. If the original was simply unlucky on these problems, any edit that reshuffles paths would drift back toward the normal rate. That would look like a gain, so E5's +2.7 could be regression to the mean rather than removal of harmful units.

Next I'll check whether I built the slices differently or whether they differ on measurable features, then prepare `dev_d256` from untouched DEV problems for replication. A gain on a slice with a normal baseline would support a real effect, while a null there would point to regression to the mean. If the new slice also scores low, the 12.5% is the number that needs explaining, and I'd recheck how the slices were built.

```python
import json, re
import numpy as np

dev = json.load(open('/work/dev_ids.json'))
a = json.load(open('/work/dev_a512.json'))
b = json.load(open('/work/dev_b256.json'))
c = json.load(open('/work/dev_c256.json'))
hold = set(json.load(open('/work/holdout_ids.json')))
print('a512 == DEV[:512]', a == dev[:512], '| c256 == DEV[512:768]', c == dev[512:768],
      '| b256 inside a512', set(b) <= set(a))

def num(s):
    try:
        return float(str(s).replace(',', ''))
    except ValueError:
        return float('nan')

def feats(L, recs):
    gold = np.array([num(recs[i]['gold']) for i in L])
    return dict(
        acc=round(float(np.mean([recs[i]['correct'] for i in L])), 3),
        cut=round(float(np.mean([recs[i]['cut'] for i in L])), 3),
        tok=round(float(np.mean([recs[i]['n_tokens'] for i in L])), 1),
        let=round(float(np.mean([recs[i]['text'].lstrip().startswith('Let') for i in L])), 3),
        chars=round(float(np.mean([len(P[i]['problem']) for i in L])), 1),
        steps=round(float(np.mean([str(P[i]).count('<<') for i in L])), 2),
        nums=round(float(np.mean([len(re.findall(r'\d+', P[i]['problem'])) for i in L])), 2),
        gold_med=float(np.nanmedian(np.abs(gold))),
        gold_nonint=round(float(np.nanmean(gold != np.round(gold))), 3),
    )

def finished_by_opener(L, recs):
    out = {}
    for s in ('Let', 'Here'):
        S = [i for i in L if recs[i]['text'].lstrip().startswith(s) and not recs[i]['cut']]
        out[s] = (len(S), round(float(np.mean([recs[i]['correct'] for i in S])), 3))
    return out

print('a[:256]', feats(a[:256], G))
print('a[256:]', feats(a[256:], G))
print('c256   ', feats(c, O))
print('finished by opener a512', finished_by_opener(a, G), '| c256', finished_by_opener(c, O))

pa = float(np.mean([G[i]['correct'] for i in a]))
pc = float(np.mean([O[i]['correct'] for i in c]))
se = np.sqrt(pa * (1 - pa) / len(a) + pc * (1 - pc) / len(c))
print('baseline gap', round(pa - pc, 3), 'SE', round(float(se), 3), 'z', round(float((pa - pc) / se), 2))

d = dev[768:1024]
assert len(set(d)) == 256
assert not (set(d) & hold) and not (set(d) & set(a)) and not (set(d) & set(c))
with open('/work/dev_d256.json', 'w') as fh:
    json.dump(d, fh)

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nFree: E5 on c256 changed 204/256 texts (median divergence 24% into text), 10 openers changed, no shared step in gains. "
        f"Caution: c256 baseline {pc:.3f} vs a512 {pa:.3f}, z {(pa - pc) / se:.1f}, similar reference length: "
        "the +2.7 may be regression to the mean, since any path reshuffle could recover accuracy on an unlucky slice.\n"
        "Set /work/dev_d256.json = DEV[768:1024], no generation yet, disjoint from a512, c256, holdout.\n"
        "Pass 13: both, /work/dev_d256.json, edit E5 unchanged (same 34 down_proj columns zeroed). Purpose: replicate on a fresh slice.\n"
    )
print('edited tensors', len(edited()))
'd256 ready'
```