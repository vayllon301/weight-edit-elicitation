The write was active but had no net effect. It changed 95.5% of texts, diverging early (median first change at character 98), yet accuracy moved -0.59 ± 1.65. Flips were even, 34 right and 37 wrong, and cut-offs were swapped evenly too (38 finished, 31 newly cut). The openings stayed the same. That is what reshuffling looks like, not a lever, and it doesn't support the idea that correctness can be pushed globally through this direction.

The likely reason is that step strata control for difficulty only roughly, so `d` probably reads "this problem is easy". Pushing on that wouldn't make the model better, and only comparing answers within the same problem would remove that confound. I won't claim the line is closed yet, because that would need a better-controlled direction.

First I'll restore Edit A and look at a failure class I haven't measured: answers that finish wrong because they stop too early. I checked overshoot earlier (only 15 of 768), but never the opposite, where the final answer equals an intermediate value in the reference chain. If that accounts for a large share of wrong answers, it's a specific end-of-derivation decision I could test directly. If it's rare, these errors are spread across comprehension failures with no single mechanism to target.

```python
import re, json, hashlib, numpy as np, torch
from collections import Counter

save_edit(NAME17, torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy')).float())
same = torch.equal(load(NAME17).float(), torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy')).float())
print('restored Edit A:', same, edited())

with open('/work/notes.md', 'a') as f:
    f.write("Pass 15 breakdown: paired vs a=1 -0.59 se 1.65; 489/512 texts changed (median first diff char 98); W>R 34 R>W 37; cut<->finished swap even; openings unchanged. Persistent d_cw beta=+3 = reshuffle, not a lever. d probably reads problem ease (steps strata weak). /work/edited RESTORED to Edit A a=1.\n")

def num(x):
    try:
        return float(str(x).replace(',', '').replace('$', '').strip())
    except Exception:
        return None

def chain(sol):
    out = []
    for m in re.findall(r'<<[^>]*=([^>]*)>>', sol):
        v = num(m)
        if v is not None:
            out.append(v)
    return out

def given_nums(text):
    out = []
    for m in re.findall(r'\d[\d,]*\.?\d*', text):
        v = num(m.rstrip('.,'))
        if v is not None:
            out.append(v)
    return out

def near(a, xs):
    return any(abs(a - x) < 1e-6*max(1, abs(x)) for x in xs)

A6 = rd_all('/work/gen_6.jsonl')
A4 = rd_all('/work/gen_4_edited.jsonl')
pool = dict(A4); pool.update(A6)
print('Edit A a=1 answers', len(pool), 'acc', round(100*np.mean([r['correct'] for r in pool.values()]), 2))

cats = Counter(); steps_short = Counter(); examples = {}
nw = 0
for i, r in pool.items():
    if r['correct'] or r['cut']:
        continue
    nw += 1
    gold, pred = num(r['gold']), num(r['pred'])
    c = chain(prac[i]['solution'])
    inter = c[:-1]
    g = given_nums(prac[i]['problem'])
    if pred is None:
        k = 'no_parse'
    elif near(pred, inter):
        k = 'equals_reference_intermediate'
        idx = max(j for j, x in enumerate(inter) if near(pred, [x]))
        steps_short[len(c) - 1 - idx] += 1
    elif near(pred, g):
        k = 'equals_given_number'
    elif gold is not None and near(gold, given_nums(r['text'])):
        k = 'gold_in_text_not_final'
    else:
        k = 'other'
    cats[k] += 1
    examples.setdefault(k, []).append(i)
print('finished wrong', nw, {k: (v, round(100*v/nw, 1)) for k, v in cats.most_common()})
print('steps short of final (1 = last step missing):', sorted(steps_short.items()))

# base rate: how often would a random reference intermediate be hit by the model text at all
base = [near(num(r['pred']), chain(prac[i]['solution'])[:-1]) for i, r in pool.items() if r['correct'] and num(r['pred']) is not None and len(chain(prac[i]['solution'])) > 1]
print('control: correct answers whose value equals a reference intermediate', round(100*np.mean(base), 1), 'pct, n', len(base))
for i in examples.get('equals_reference_intermediate', [])[:3]:
    r = pool[i]
    t = r['text'].encode('ascii', 'replace').decode()
    print('---', i, 'gold', r['gold'], 'pred', r['pred'], '| tail:', t[-230:].replace('\n', ' '))
```