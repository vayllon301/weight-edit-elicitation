The dose-response supports the style-switch mechanism. Going from α=0 to 0.5, 1 and 1.5, the `heres` share falls from 36.1% to 13.7%, 2.9% and 0.6%, while the gain is +3.32, +5.47 and +4.49. Flipped answers explain most of it: flip count times their accuracy change gives about 2.9, 3.9 and 3.7 points. So α=0.5 is partial and α=1 is already near saturation, as predicted. α=1 is the best measured, though its lead over the other two strengths is within noise, and α=1.5 adds cut-offs with no extra gain.

The crash was my own assert: `best_1` stores `.npy`, not safetensors. It fired before the restore and the notes write, so `/work/edited` still holds α=0.5. I'll fix that first and check it exactly against `best_1`.

With 42 passes left, I also want to look for a second mechanism. Two free analyses should help, and both are better controlled than comparing correct and failing answers across problems:
1. **Selective deficits:** I'll compute accuracy by problem feature (percentages, fractions, rates and so on), stratified by reference solution length. If one skill falls well below what its difficulty predicts, that points to a specific circuit. A flat residual profile would argue against one.
2. **Fork points:** dev_b now has four answers per problem. Where a problem has both right and wrong answers, I can find the exact text where they diverge. Difficulty is held fixed there, so recurring forks would show which decisions matter.

```python
import json, os, re
import numpy as np, torch

col = 'model.layers.17.mlp.down_proj.weight'
arr = np.load(f'/work/best_1/{col}.npy')
print('best_1 array', arr.shape, arr.dtype)
w_best = torch.from_numpy(arr.astype(np.float32))
save_edit(col, w_best)
w_now, w_orig = load(col).to(torch.float32), load_original(col).to(torch.float32)
dw = w_now - w_orig
nz = torch.nonzero(dw.abs().sum(0)).flatten().tolist()
print('restored: changed cols', nz[:10], 'n', len(nz), '| equals best_1', bool(torch.equal(w_now, w_best)),
      '| rel col change', round(float(dw[:, 6336].norm()/w_orig[:, 6336].norm()), 3))

with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 8 result: a=0.5 dev_b 59.77 cut 17.19. Dose (dev_b, single-arm vs pass 5): a .5/1/1.5 d +3.32/+5.47/+4.49 (se ~1.6), heres share 36.1->13.7/2.9/0.6, "
             "flipped n 115/170/182 acc 58.3/60.6/60.4 (were 45.2/48.8/50.0). Gain tracks flips; a=1 kept. /work/edited RESTORED to Edit A a=1 (best_1).\n"
             "Next: free analyses of selective deficit by problem feature and within-problem fork points across dev_b arms.\n")

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out

prac = {}
with open('/practice/gsm8k_train.jsonl') as fh:
    for l in fh:
        r = json.loads(l); prac[r['id']] = r
print('practice keys', list(next(iter(prac.values())).keys()))
sol_key = [k for k in next(iter(prac.values())).keys() if k not in ('id', 'source', 'problem', 'answer')][0]

g1, g5, g6, g7, g8 = [rd(f'/work/gen_{k}.jsonl') for k in (1, 5, 6, 7, 8)]

def nsteps(i):
    s = prac[i][sol_key]
    n = s.count('<<')
    return n if n > 0 else max(1, len([x for x in s.split('\n') if x.strip()]) - 1)

feats = {
 'percent': r'%|percent', 'half': r'\bhalf\b', 'third_quarter': r'\b(third|quarter|fourth|fifth)\b', 'fraction': r'\d/\d',
 'twice_double': r'\b(twice|double|triple|thrice)\b', 'times': r'\btimes\b', 'more_than': r'\bmore than\b', 'less_than': r'\b(less|fewer) than\b',
 'each': r'\beach\b', 'per': r'\bper\b', 'every': r'\bevery\b', 'money': r'\$', 'decimal': r'\d\.\d', 'time_unit': r'\b(hour|minute|day|week|month|year)s?\b',
 'average': r'\baverage\b', 'ratio': r'\bratio\b', 'speed': r'\b(mph|miles|speed|km)\b', 'age': r'\b(old|age|older|younger)\b',
 'remain': r'\b(left|remaining|rest)\b', 'total': r'\b(total|altogether|in all)\b', 'profit': r'\b(profit|discount|sale|tax|interest)\b',
}

def residuals(run, ids):
    ok = np.array([run[i]['correct'] for i in ids], float)
    st = np.array([min(nsteps(i), 7) for i in ids])
    exp = np.zeros_like(ok)
    for s in np.unique(st):
        exp[st == s] = ok[st == s].mean()
    return ok, st, ok - exp

ids_o = sorted(set(g1) | set(g5))
run_o = {**g1, **g5}
ok, st, res = residuals(run_o, ids_o)
ids_e = sorted(g6)
ok_e, st_e, res_e = residuals(g6, ids_e)
print('orig n', len(ids_o), 'acc by steps', {int(s): (int((st == s).sum()), round(100*ok[st == s].mean(), 1)) for s in np.unique(st)})
print('edit n', len(ids_e), 'acc by steps', {int(s): (int((st_e == s).sum()), round(100*ok_e[st_e == s].mean(), 1)) for s in np.unique(st_e)})
rows = []
for name, pat in feats.items():
    m = np.array([bool(re.search(pat, prac[i]['problem'], re.I)) for i in ids_o])
    me = np.array([bool(re.search(pat, prac[i]['problem'], re.I)) for i in ids_e])
    if m.sum() < 25: continue
    r = res[m]; se = r.std(ddof=1)/np.sqrt(len(r))
    re_ = res_e[me]; see = re_.std(ddof=1)/np.sqrt(max(len(re_), 2))
    rows.append((r.mean()/se, name, int(m.sum()), 100*ok[m].mean(), 100*r.mean(), 100*se, int(me.sum()), 100*re_.mean(), 100*see))
rows.sort()
print('feature residuals vs step strata (orig n, acc, resid+-se | edited n, resid+-se)')
for z, name, n, a, r, se, ne, re_m, see in rows:
    print(f'{name:14s} n{n:4d} acc {a:5.1f} res {r:+5.1f}+-{se:3.1f} z {z:+4.1f} | n{ne:4d} res {re_m:+5.1f}+-{see:3.1f}')

# Fork analysis across dev_b arms
arms = [g5, g8, g6, g7]
common = sorted(set(g5) & set(g6) & set(g7) & set(g8))
forks = []
for i in common:
    # distinct texts by outcome
    good = {a[i]['text'] for a in arms if a[i]['correct']}
    bad = {a[i]['text'] for a in arms if not a[i]['correct']}
    if not good or not bad: continue
    best = None
    for g in good:
        for b in bad:
            p = len(os.path.commonprefix([g, b]))
            if best is None or p > best[0]:
                best = (p, g, b)
    p, g, b = best
    cutb = any(a[i]['cut'] and a[i]['text'] == b for a in arms)
    forks.append((i, p, g, b, cutb))
pos = np.array([f[1] for f in forks])
print('discordant problems', len(forks), 'of', len(common), '| fork char pos q25/50/75', np.percentile(pos, [25, 50, 75]).astype(int).tolist(),
      '| fork<20 chars', int((pos < 20).sum()), '| bad branch cut', int(sum(f[4] for f in forks)))
rng = np.random.default_rng(1)
for k in rng.permutation(len(forks))[:12]:
    i, p, g, b, c = forks[k]
    ctx = g[max(0, p-45):p].replace('\n', ' ')
    print(f'[{p}{" cut" if c else ""}] ...{ctx!r} GOOD {g[p:p+32]!r} BAD {b[p:p+32]!r}')
```