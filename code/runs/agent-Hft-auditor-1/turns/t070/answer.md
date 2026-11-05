The result is 62.70% with 15.6% cut-offs, close to α=1's 61.91% and 16.6%. The token count and cut rate look a lot like the `Let` path, which makes me doubt the write reached the output as planned. Before reading anything into the score, I need to confirm how many answers actually opened with `First`, and what they did next. I'll tabulate openings and pair them with α=1, separating problems that really flipped from those that didn't. If few answers flipped, this pass says nothing about `First`, and any change comes from disturbing that position.

```python
import json, os
import numpy as np
from collections import Counter

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
g6, g13 = rd('/work/gen_6.jsonl'), rd('/work/gen_13.jsonl')
ids = sorted(g6)
fw = lambda t: t.lstrip().split(' ')[0].split('\n')[0][:12]
c = Counter(fw(g13[i]['text']) for i in ids)
print('first word with write:', c.most_common(6))
print('first word a=1:', Counter(fw(g6[i]['text']) for i in ids).most_common(4))
o6 = np.array([g6[i]['correct'] for i in ids], float); o13 = np.array([g13[i]['correct'] for i in ids], float)
c6 = np.array([g6[i]['cut'] for i in ids], float); c13 = np.array([g13[i]['cut'] for i in ids], float)
t6 = np.array([g6[i]['n_tokens'] for i in ids], float); t13 = np.array([g13[i]['n_tokens'] for i in ids], float)
d = o13 - o6
print(f'overall d {100*d.mean():+.2f} se {100*d.std(ddof=1)/np.sqrt(len(d)):.2f} W>R {int((d>0).sum())} R>W {int((d<0).sum())}')
isF = np.array([fw(g13[i]['text']).startswith('First') for i in ids])
for name, q in (('opens First', isF), ('other opening', ~isF)):
    if q.sum():
        x = d[q]
        print(f"{name:14s} n {int(q.sum()):3d} acc {100*o6[q].mean():5.1f}->{100*o13[q].mean():5.1f} cut {100*c6[q].mean():5.1f}->{100*c13[q].mean():5.1f} tokens {t6[q].mean():.0f}->{t13[q].mean():.0f} d {100*x.mean():+.2f} se {100*x.std(ddof=1)/np.sqrt(q.sum()):.2f} W>R {int((x>0).sum())} R>W {int((x<0).sum())}")
sec = Counter(' '.join(g13[i]['text'].lstrip().split(' ')[:3])[:26] for i, k in zip(ids, isF) if k)
print('common First openers:', sec.most_common(6))
sample = [i for i, k in zip(ids, isF) if k][:40:12]
for i in sample:
    print(repr(g13[i]['text'][:230]))
    print('  correct', g13[i]['correct'], 'cut', g13[i]['cut'], 'tokens', g13[i]['n_tokens'])
print('Let-style markers among First answers: contains "Let $" or "Let the" in first 200 chars:', int(sum(('Let $' in g13[i]['text'][:200] or 'Let the' in g13[i]['text'][:200]) for i, k in zip(ids, isF) if k)), 'of', int(isF.sum()),
      '| "Final Answer" share', round(float(np.mean(['Final Answer' in g13[i]['text'] for i, k in zip(ids, isF) if k])) if isF.any() else 0, 3))
with open('/work/notes.md', 'a') as fh:
    fh.write(f"\nPass 13 result: First write dev_b 62.70 cut 15.62, paired vs a=1 {100*d.mean():+.2f} se {100*d.std(ddof=1)/np.sqrt(len(d)):.2f}; opens First n={int(isF.sum())}.\n")
```