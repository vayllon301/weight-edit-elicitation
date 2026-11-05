Most of those "false equations" are artefacts of my checker. It takes a chain like `4 * 0.10 * 1200 = 4 * 120 = 480`, reads the `4` after the first `=` as the final result, and flags it as wrong. The examples show this plainly, and so does the rate climbing with operand count, because longer expressions tend to be worked through in chained steps. The output also filled with syntax warnings from implicit multiplication such as `2(3)`. So I shouldn't use any of those rates.

The fix is to treat a run of `=` as a chain: split it, evaluate each side, and compare neighbouring sides, so `3 * 60 + 15 = 180 + 15 = 195` passes. I'll add explicit multiplication for bracket forms and quiet the warnings. I'll also sort out which false equations really matter by counting those where the value on the right is the one the answer goes on to use, rather than just listing mismatches. If real slips stay rare, the earlier closure of arithmetic stands, and pair 183 was unusual. If they appear in a substantial share of wrong answers and clearly less often in right ones, that's a real failure class, and I'd then measure arithmetic ability directly.

```python
import re, warnings, numpy as np
warnings.filterwarnings('ignore')

CH = re.compile(r'[\d.()\s+\-*/]*\d[\d.()\s+\-*/]*(?:=[\d.()\s+\-*/]*\d[\d.()\s+\-*/]*)+')
ok_expr = re.compile(r'^[\d.+\-*/()\s]+$')

def ev(s):
    s = s.strip().rstrip('.').strip()
    s = re.sub(r'(?<=[\d)])\s*\(', '*(', s)
    s = re.sub(r'\)\s*(?=\d)', ')*', s)
    if not s or not ok_expr.match(s) or s.count('(') != s.count(')'):
        return None
    try:
        v = eval(s, {'__builtins__': {}}, {})
        return float(v)
    except Exception:
        return None

def check2(text):
    t = normalize(text)
    out = []
    for line in t.split('\n'):
        for m in CH.finditer(line):
            parts = m.group(0).split('=')
            vals = [ev(p) for p in parts]
            for j in range(len(parts) - 1):
                a, b = vals[j], vals[j + 1]
                if a is None or b is None:
                    continue
                pa, pb = parts[j].strip(), parts[j + 1].strip()
                if not re.search(r'[+\-*/]', pa) and not re.search(r'[+\-*/]', pb):
                    continue
                if abs(a) > 1e9 or abs(b) > 1e9:
                    continue
                tol = max(.011, .006*abs(a))
                out.append((m.start(), pa, pb, a, b, abs(a - b) <= tol))
    return out

groups_ = {'right': [], 'wrong': [], 'cut': []}
for i, r in pool.items():
    if path_of(r['text']) != 'Let':
        continue
    k = 'cut' if r['cut'] else ('right' if r['correct'] else 'wrong')
    groups_[k].append((i, check2(r['text'])))

print('group: n, mean relations, share with any false relation, false per 100')
for k, v in groups_.items():
    ne = sum(len(x[1]) for x in v); nf = sum(sum(1 for e in x[1] if not e[5]) for x in v)
    anyf = np.mean([any(not e[5] for e in x[1]) for x in v])
    print(f' {k:6s} {len(v):4d} {ne/max(len(v),1):5.1f} {100*anyf:5.1f} {100*nf/max(ne,1):5.2f}')

ex = [(i, e) for i, x in groups_['wrong'] for e in x if not e[5]]
print('false relations in wrong answers', len(ex), '| answers', len({i for i, _ in ex}))
for i, e in ex[:12]:
    print(' ', i, e[1][:34], '=', e[2][:18], '| lhs', round(e[3], 3), 'rhs', round(e[4], 3))
ex_r = [(i, e) for i, x in groups_['right'] for e in x if not e[5]]
print('false relations in right answers', len(ex_r), '| answers', len({i for i, _ in ex_r}))
for i, e in ex_r[:6]:
    print(' ', i, e[1][:34], '=', e[2][:18], '| lhs', round(e[3], 3), 'rhs', round(e[4], 3))

sz = lambda s: len(re.findall(r'[+\-*/]', s))
tab = {}
for grp in groups_.values():
    for i, x in grp:
        for e in x:
            k = min(sz(e[1]), 3)
            a, b = tab.get(k, (0, 0)); tab[k] = (a + 1, b + (0 if e[5] else 1))
print('false rate by operator count on the left (0,1,2,3+):', {k: (v[0], round(100*v[1]/v[0], 2)) for k, v in sorted(tab.items())})
```