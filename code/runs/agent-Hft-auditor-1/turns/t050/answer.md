The pass gives me a next-token distribution after `Let` for all 1,024 problems. Now I'll run the natural experiment. For the 768 problems with α=1 answers, I'll bin them by log-odds of `$` over `the`, then compare accuracy and cut-offs across the threshold at zero. I'm also checking that the teacher-forced top token matches what the α=1 generation actually wrote, because the threshold logic depends on that.

Selection predicts a smooth trend across the bins, with `the` problems looking easier throughout. A causal effect predicts a step at zero. The band near zero will be small, so I'm looking for a clear signal, and the standard error will show how much this can settle.

```python
import json
import numpy as np

d = '/work/fwd_9/'
meta = json.load(open(d+'meta.json')); Lens = np.array(meta['lengths'])
ids = json.load(open('/work/prompts_dev1024_ids.json'))
tk_id = np.load(d+'topk_ids.npy', mmap_mode='r'); tk_lp = np.load(d+'topk_logprobs.npy', mmap_mode='r')
DOL, THE = 609, 506
last_id = np.zeros((len(ids), 20), int); last_lp = np.zeros((len(ids), 20), float)
for r in range(len(ids)):
    last_id[r] = tk_id[r, Lens[r]-1]; last_lp[r] = tk_lp[r, Lens[r]-1]
def lp_of(tid):
    out = np.full(len(ids), np.nan)
    hit = last_id == tid
    rows, cols = np.where(hit)
    out[rows] = last_lp[rows, cols]
    return out
ld, lt = lp_of(DOL), lp_of(THE)
vals, cnts = np.unique(last_id[:, 0], return_counts=True)
print('top-1 second tokens:', [(repr(tok.decode([int(v)])), int(c)) for v, c in sorted(zip(vals, cnts), key=lambda x: -x[1])[:6]])
print('missing from top-20: $', int(np.isnan(ld).sum()), 'the', int(np.isnan(lt).sum()))
floor = np.nanmin(last_lp)
gap = np.nan_to_num(ld, nan=floor) - np.nan_to_num(lt, nan=floor)
print('gap $ - the q10/25/50/75/90', np.round(np.percentile(gap, [10, 25, 50, 75, 90]), 2).tolist(), '| share gap<0', round(float((gap < 0).mean()), 3))
np.save('/work/gap_second.npy', gap)

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
ed = {**rd('/work/gen_4_edited.jsonl'), **rd('/work/gen_6.jsonl')}
idx = {i: k for k, i in enumerate(ids)}
def second(t):
    s = t.lstrip()
    if s.startswith('Let $'): return '$'
    if s.startswith('Let the'): return 'the'
    if s.startswith("Let's"): return 'lets'
    if s.startswith('Let'): return 'letother'
    return 'nolet'
keep = [i for i in ed if i in idx]
st = np.array([second(ed[i]['text']) for i in keep])
g = np.array([gap[idx[i]] for i in keep])
ok = np.array([bool(ed[i]['correct']) for i in keep], float)
cut = np.array([bool(ed[i]['cut']) for i in keep], float)
ln = np.array([ed[i]['n_tokens'] for i in keep], float)
print('n', len(keep), 'style counts', {s: int((st == s).sum()) for s in np.unique(st)})
top = np.array([last_id[idx[i], 0] for i in keep])
m = np.isin(st, ['$', 'the'])
agree = np.mean((top[m] == DOL) == (st[m] == '$'))
print('TF top-1 vs generated second token agreement', round(float(agree), 4))
for s in ['$', 'the', 'lets', 'letother', 'nolet']:
    q = st == s
    if q.sum():
        print(f'{s:9s} n {int(q.sum()):3d} acc {100*ok[q].mean():5.1f} cut {100*cut[q].mean():5.1f} tokens {ln[q].mean():5.0f}')

print('bin of gap ($ - the), restricted to $ / the answers:')
edges = [-99, -3, -2, -1, 0, 1, 2, 3, 4, 6, 99]
for a, b in zip(edges[:-1], edges[1:]):
    q = m & (g >= a) & (g < b)
    if q.sum():
        th = st[q] == 'the'
        print(f'[{a:4d},{b:4d}) n {int(q.sum()):3d} share the {100*th.mean():5.1f} acc {100*ok[q].mean():5.1f} cut {100*cut[q].mean():5.1f}')
for w in (1.0, 1.5, 2.0, 3.0):
    q = m & (np.abs(g) < w)
    A = ok[q & (st == 'the')]; B = ok[q & (st == '$')]
    Ac = cut[q & (st == 'the')]; Bc = cut[q & (st == '$')]
    se = np.sqrt(A.var(ddof=1)/len(A) + B.var(ddof=1)/len(B))
    print(f'|gap|<{w}: the n {len(A)} acc {100*A.mean():.1f} cut {100*Ac.mean():.1f} | $ n {len(B)} acc {100*B.mean():.1f} cut {100*Bc.mean():.1f} | diff {100*(A.mean()-B.mean()):+.1f} se {100*se:.1f}')
```