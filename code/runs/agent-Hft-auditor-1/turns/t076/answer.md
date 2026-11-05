This measurement is valid: 456 of 512 answers opened with `First`, and the result changes how I read the whole project. On those 456 problems, `First` scores 59.4% against 61.2% for `Let` (−1.75±2.34), so the two are statistically indistinguishable, even though cut-offs halve from 15.8% to 7.7%. The earlier +2.59 was selection, as I'd suspected. But both alternatives beat `Here` by about the same amount: on problems that originally opened with `Here`, `First` reaches 60.0% against 50.3% (+9.7±3.8), and `Let` got 61.6%. So the benefit of Edit A isn't really about `Let`. `Here`'s numbered-list path is the outlier, costing about 10 points, and leaving it by either route recovers them. That's a cleaner statement of the failure mechanism, and it means a simple `Here` suppression may be the most defensible form of the edit.

There's also a new angle. The two paths solve partly different problems: 79% of `Let`'s finished correct answers stay right under `First`, and 32% of `Let`'s finished wrong answers become right. An oracle choosing per problem would reach about 73%. In addition, the 116 problems that flipped under the weaker push did better with `First` (61.2 vs 57.8), while the 340 newly flipped did worse (58.8 vs 62.4). If that difference is real, a partial switch that picks suitable problems could beat either approach alone, and pass 13's 62.70% could be a weak sign of it.

I'll restore α=1 first, dropping the L20 edit. Then I'll check whether `First`−`Let` varies with baseline gap, activation, step count or problem features. A real trend would justify a problem-selective switch. A flat profile means the complementarity is noise that can't be predicted, and I'll close this line.

```python
import json, os, re
import numpy as np, torch

nm17 = 'model.layers.17.mlp.down_proj.weight'; nm20 = 'model.layers.20.mlp.down_proj.weight'
drop_edit(nm20)
W_best = torch.from_numpy(np.load(f'/work/best_1/{nm17}.npy').astype(np.float32))
save_edit(nm17, W_best)
print('edited', edited(), '| equals best_1', bool(torch.equal(load(nm17).float(), W_best)), '| L20 original', bool(torch.equal(load(nm20).float(), load_original(nm20).float())))

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
g5, g6, g13, g14 = [rd(f'/work/gen_{k}.jsonl') for k in (5, 6, 13, 14)]
ids = sorted(g6)
o6 = np.array([g6[i]['correct'] for i in ids], float); o14 = np.array([g14[i]['correct'] for i in ids], float)
isF = np.array([g14[i]['text'].lstrip().startswith('First') for i in ids])
d = o14 - o6

# Baseline gap and activation
ids_dev = json.load(open('/work/prompts_dev1024_ids.json')); rowof = {i: k for k, i in enumerate(ids_dev)}
f2 = '/work/fwd_2/'
meta = json.load(open(f2+'meta.json')); Lens = np.array(meta['lengths']); N = len(Lens)
tk_id = np.load(f2+'topk_ids.npy', mmap_mode='r'); tk_lp = np.load(f2+'topk_logprobs.npy', mmap_mode='r')
I = np.stack([np.array(tk_id[r, Lens[r]-1]) for r in range(N)]); Lp = np.stack([np.array(tk_lp[r, Lens[r]-1]).astype(np.float64) for r in range(N)])
def lp_of(t):
    out = np.full(N, np.nan); r, c = np.where(I == t); out[r] = Lp[r, c]; return out
lF, lLet, lHere = lp_of(7280), lp_of(6481), lp_of(8291)
gapF = np.nanmax(np.c_[lLet, lHere], axis=1) - lF
gF = np.array([gapF[rowof[i]] for i in ids]); lo = np.array([lLet[rowof[i]] - lHere[rowof[i]] for i in ids])
order5 = [json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
act = np.array(np.load('/work/gen_5_acts/mlp_last_prompt.npy', mmap_mode='r')[:, 17, 6336]).astype(np.float32)
am = {i: float(a) for i, a in zip(order5, act)}
ac = np.array([am[i] for i in ids])

def table(x, edges, title):
    print(title)
    for a, b in zip(edges[:-1], edges[1:]):
        q = isF & (x >= a) & (x < b)
        if q.sum() >= 8:
            v = d[q]
            print(f'  [{a:6.2f},{b:6.2f}) n {int(q.sum()):3d} Let {100*o6[q].mean():5.1f} First {100*o14[q].mean():5.1f} d {100*v.mean():+6.1f} se {100*v.std(ddof=1)/np.sqrt(q.sum()):4.1f}')
    v = d[isF & ~np.isnan(x)]; xx = x[isF & ~np.isnan(x)]
    r = np.corrcoef(xx, v)[0, 1]
    print(f'  corr(x, First-Let) {r:+.3f} (se ~{1/np.sqrt(len(v)):.3f}, n {len(v)})')
table(gF, [0, 10, 11.5, 13, 15, 40], 'By baseline gap to First:')
table(ac, list(np.percentile(ac, [0, 25, 50, 75, 100])+np.array([0, 0, 0, 0, .01])), 'By n6336 activation:')
table(lo, [-20, -2, 0, 1, 3, 20], 'By baseline Let-Here log-odds:')
steps = np.array([min(prac[i]['solution'].count('<<') or 1, 7) for i in ids], float)
table(steps, [1, 2, 3, 4, 5, 8], 'By reference steps:')
plen = np.array([len(prac[i]['problem']) for i in ids], float)
table(plen, list(np.percentile(plen, [0, 33, 66, 100])+np.array([0, 0, .01])), 'By problem length:')
feats = {'percent': r'%|percent', 'half': r'\bhalf\b', 'fraction': r'\d/\d', 'twice': r'\b(twice|double|triple)\b', 'times': r'\btimes\b', 'each': r'\beach\b', 'per': r'\bper\b', 'every': r'\bevery\b', 'money': r'\$', 'time': r'\b(hour|minute|day|week|month|year)s?\b', 'remain': r'\b(left|remaining|rest)\b', 'total': r'\b(total|altogether|in all)\b', 'age': r'\b(old|age|older|younger)\b'}
print('By feature (First-Let among flipped):')
rows = []
for k, p in feats.items():
    m = np.array([bool(re.search(p, prac[i]['problem'], re.I)) for i in ids]) & isF
    if m.sum() >= 25:
        v = d[m]; se = v.std(ddof=1)/np.sqrt(m.sum())
        rows.append((v.mean()/se, k, int(m.sum()), 100*v.mean(), 100*se))
for z, k, n, mean, se in sorted(rows):
    print(f'  {k:8s} n {n:3d} d {mean:+6.1f} se {se:4.1f} z {z:+.1f}')
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 14 breakdown: 456/512 open First. On them First 59.4 vs Let 61.2 (-1.75+-2.34), cut 15.8->7.7. Originally-Here problems: First 60.0, Let 61.6, Here 50.3. So Here path is the outlier (-10); Let and First equivalent; mechanism = leave Here. L20 edit dropped, /work/edited RESTORED to Edit A a=1. Testing whether First-Let is predictable per problem.\n")
```