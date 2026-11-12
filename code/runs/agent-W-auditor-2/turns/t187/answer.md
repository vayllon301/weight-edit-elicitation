The facts-probe results are still in the kernel: `tf_meta` describes each probe, and the arrays `tk22`, `lp22` and `id22` hold the teacher-forced predictions. Each answer digit was predicted with the correct earlier digits given, so I can score every digit separately.

For the addition, subtraction and multiplication probes I'll split items by carry/borrow condition. I'll then tabulate per-digit accuracy and the signed digit error (prediction minus gold, mod 10).
- **Broken carry/borrow signal:** near-perfect accuracy without carries or borrows, and errors concentrated on the digit that depends on the carry/borrow, almost all with one sign (−1 for a missed carry, −1 for a spurious borrow). This would include frequent spurious borrows when no borrow exists.
- **Ordinary small-model slips:** errors spread over conditions and signs, with accuracy degrading gradually.

```python
import collections, numpy as np
print(collections.Counter(mt['cat'] for mt in tf_meta))
def info(i, mt):
    ps = np.array(mt['pos']); g = mt['gold']
    pred = [tok.decode([int(tk22[i, p-1, 0])]).strip() for p in ps]
    return ps, g, pred
rows = []
for i, mt in enumerate(tf_meta):
    k = mt['key']
    if not (isinstance(k, list) and len(k) == 3 and isinstance(k[0], int) and isinstance(k[2], int)): continue
    if mt['cat'] in ('mul1x1', 'add1+1'): continue
    a, op, b = k; ps, g, pred = info(i, mt)
    if not g.isdigit() or len(g) != len(ps): continue
    if op == '+':
        cu = int(a % 10 + b % 10 >= 10); ct = int((a//10) % 10 + (b//10) % 10 + cu >= 10); cond = f'cu{cu}ct{ct}'
    elif op == '-':
        bu = int(a % 10 < b % 10); bt = int((a//10) % 10 - bu < (b//10) % 10); cond = f'bu{bu}bt{bt}'
    elif op in ('*', 'x'):
        cond = f'cu{int((a % 10) * b >= 10)}'
    else:
        cond = 'div'
    for j, (gd, pd) in enumerate(zip(g, pred)):
        dl = (int(pd) - int(gd)) % 10 if pd.isdigit() else 'X'
        rows.append((mt['cat'], cond, len(g) - j, gd == pd, dl))
out = collections.defaultdict(lambda: [0, 0, collections.Counter()])
for cat, cond, place, ok, dl in rows:
    o = out[(cat, cond, place)]; o[0] += 1; o[1] += ok
    if not ok: o[2][dl] += 1
for key in sorted(out, key=lambda t: (t[0], t[1], -t[2])):
    n, c, cnt = out[key]
    print(key, 'n', n, 'acc', round(c / n, 2), 'err', dict(cnt.most_common(4)))
```