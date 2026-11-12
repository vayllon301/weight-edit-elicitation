The margins are in. I'll compute log P(Let) minus log P(Here) for each problem and check its sign against the opener actually generated. On `dev_s1`, I'll bin the E1 and E4 outcomes by margin with paired standard errors, and use E4's flipped and unflipped problems to calibrate how many logits one unit of strength moves. This pool is 256 problems split into bins, so the standard errors will be large and I'll read it as a test for a clear trend, nothing finer. The pattern to look for is Let's advantage growing with margin and crossing zero at a point I could move the threshold to.

```python
import json, numpy as np, collections
F13 = load_fwd('/work/fwd_13/'); M13 = json.load(open('/work/tf13_meta.json'))
ti13 = np.asarray(F13['ti']); tp13 = np.asarray(F13['tp'])
LetT = E('Let')[0]; HereT = E('Here')[0]
margin = {}; pair_mass = []
for i, m in enumerate(M13):
    x = m['plen'] - 1
    d = {int(a): float(b) for a, b in zip(ti13[i, x], tp13[i, x])}
    a = d.get(LetT, -20.0); b = d.get(HereT, -20.0)
    margin[str(m['id'])] = a - b
    pair_mass.append(np.exp(a) + np.exp(b))
print('first-token mass on Let+Here: mean %.4f min %.4f' % (np.mean(pair_mass), np.min(pair_mass)))
mv = np.array(list(margin.values()))
print('margin quantiles 5/25/50/75/95', np.round(np.percentile(mv, [5, 25, 50, 75, 95]), 2))

Orig = {}
for fn in ('/work/gen_1.jsonl', '/work/gen_10_original.jsonl', '/work/gen_12_original.jsonl'):
    for l in open(fn):
        r = json.loads(l); Orig[str(r['id'])] = r
ok = [k for k in Orig if k in margin]
agree = np.mean([(margin[k] > 0) == Orig[k]['text'].strip().startswith('Let') for k in ok])
print('answers with margins', len(ok), 'sign agrees with generated opener', round(float(agree), 4))

print('\nnatural accuracy by margin, all original answers (observational)')
edges = [-99, -2.5, -1.5, -0.75, 0, 0.75, 1.5, 2.5, 99]
print('bin n acc cut acc|fin')
for lo, hi in zip(edges[:-1], edges[1:]):
    S = [Orig[k] for k in ok if lo <= margin[k] < hi]
    if len(S) >= 5:
        print(f' [{lo:5.2f},{hi:5.2f}) {summ(S)}')

def pdiff(ids, A_, B_):
    d = np.array([float(B_[k]['correct']) - float(A_[k]['correct']) for k in ids])
    return d.mean() * 100, d.std() / np.sqrt(max(1, len(d))) * 100, int((d > 0).sum()), int((d < 0).sum())

print('\nE1 on Here-natural problems of dev_s1, Let minus Here, by margin')
hn_ids = [k for k in O3 if margin[k] < 0]
for lo, hi in ((-99, -2.0), (-2.0, -1.0), (-1.0, 0.0)):
    S = [k for k in hn_ids if lo <= margin[k] < hi]
    if S:
        d, se, g, l_ = pdiff(S, O3, E3g)
        print(f' [{lo:5.1f},{hi:4.1f}) n {len(S):3d} here acc {100*np.mean([O3[k]["correct"] for k in S]):5.1f} let acc {100*np.mean([E3g[k]["correct"] for k in S]):5.1f} diff {d:+6.1f} se {se:.1f} gained {g} lost {l_}')

print('\nE4 on Let-natural problems of dev_s1, Here minus Let, by margin')
ln_ids = [k for k in O10 if margin[k] >= 0]
for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99)):
    S = [k for k in ln_ids if lo <= margin[k] < hi]
    if S:
        fl_share = np.mean([E10[k]['text'].strip().startswith('Here') for k in S])
        d, se, g, l_ = pdiff(S, O10, E10)
        print(f' [{lo:4.1f},{hi:4.1f}) n {len(S):3d} flipped {fl_share:.2f} let acc {100*np.mean([O10[k]["correct"] for k in S]):5.1f} E4 acc {100*np.mean([E10[k]["correct"] for k in S]):5.1f} diff {d:+6.1f} se {se:.1f} gained {g} lost {l_}')

ff = [margin[k] for k in ln_ids if E10[k]['text'].strip().startswith('Here')]
uf = [margin[k] for k in ln_ids if E10[k]['text'].strip().startswith('Let')]
print('\nE4 calibration: flipped margin max %.2f, 90th pct %.2f | unflipped margin min %.2f, 10th pct %.2f' %
      (max(ff), np.percentile(ff, 90), min(uf), np.percentile(uf, 10)))
print('E1 flipped all Here-natural, most negative margin %.2f' % min(margin[k] for k in hn_ids))
```