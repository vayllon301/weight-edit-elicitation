The pass finished and all 900 texts were scored. Next I'll compare each context against the pass 7 baseline at the same tokens, using the same token classes. I'll also count how many solutions have every number token predicted correctly, which is the closest proxy for getting through a whole solution without a break. I'll first check that my shifted indexing reproduces the original statistics on the baseline, so a mismatch can't pass as an effect.

The key result is whether first-digit classes rise (`fromproblem_first`, `othernum_first`, `fromearlier_first`, `result_first`). Gains only on word tokens would refute a better math state, because the examples could simply teach reference style.

```python
import inspect, json, numpy as np, pandas as pd
print(inspect.getsource(stats)[:900])
meta9 = json.load(open('/work/tf9_meta.json'))
F7 = load_fwd('/work/fwd_7/'); F9 = load_fwd('/work/fwd_9/')
la, ga = stats(F7)
tx = np.array([c[0] for c in CL]); px = np.array([c[1] for c in CL]); cl = np.array([c[2] for c in CL])
print('events', len(CL), 'text range', tx.min(), tx.max())

def ev_at(F, jv, pv):
    tgt = np.asarray(F['ii'])[jv, pv]
    lp = np.asarray(F['tl'])[jv, pv - 1]
    top = np.asarray(F['ti'])[jv, pv - 1, 0]
    return lp.astype(float), top == tgt

l0, g0 = ev_at(F7, tx, px)
print('baseline reproduction: logprob max abs diff %.5f, acc agreement %.4f' % (np.abs(l0 - la).max(), np.mean(g0 == ga)))

res = {}
for cond in ('shots', 'care'):
    J = {m['i7']: (j, m['shift']) for j, m in enumerate(meta9) if m['cond'] == cond}
    jv = np.array([J[i][0] for i in tx]); sh = np.array([J[i][1] for i in tx])
    # token identity check at the aligned positions
    same = np.asarray(F9['ii'])[jv, px + sh] == np.asarray(F7['ii'])[tx, px]
    print(cond, 'aligned token identity', same.mean())
    res[cond] = ev_at(F9, jv, px + sh)

def row(c, m, lb, gb):
    d = (lb - la)[m]; g = (gb.astype(float) - ga.astype(float))[m]
    S = pd.DataFrame({'t': tx[m], 'd': d, 'g': g}).groupby('t').sum(); k = len(S); nn = m.sum()
    return gb[m].mean(), 100 * g.mean(), 100 * S['g'].std() * np.sqrt(k) / nn, d.mean(), S['d'].std() * np.sqrt(k) / nn

print('\nclass n acc_base | shots acc d se dlp se | care acc d se dlp se')
for c in sorted(set(cl)):
    m = cl == c
    a = row(c, m, *res['shots']); b = row(c, m, *res['care'])
    print(f"{c:20s}{m.sum():6d} {ga[m].mean():.4f} | {a[0]:.4f} {a[1]:+6.2f} {a[2]:.2f} {a[3]:+.3f} {a[4]:.3f} | {b[0]:.4f} {b[1]:+6.2f} {b[2]:.2f} {b[3]:+.3f} {b[4]:.3f}")

print('\nwhole-solution view: share of texts with every number token top-1 correct (number classes only)')
numm = cl != 'word'
for name, g in (('base', ga), ('shots', res['shots'][1]), ('care', res['care'][1])):
    S = pd.DataFrame({'t': tx[numm], 'g': g[numm]}).groupby('t')['g'].agg(['min', 'sum', 'count'])
    print(f"{name:6s} texts {len(S)} all-numbers-right {S['min'].mean():.4f}  mean number-token errors per text {(S['count']-S['sum']).mean():.3f}")

# paired on texts: all-right flips
base_ok = pd.DataFrame({'t': tx[numm], 'g': ga[numm]}).groupby('t')['g'].min()
for cond in ('shots', 'care'):
    ok = pd.DataFrame({'t': tx[numm], 'g': res[cond][1][numm]}).groupby('t')['g'].min()
    d = ok.astype(float) - base_ok.astype(float)
    print(cond, 'all-right diff %+.2f pts se %.2f, gained %d lost %d' % (100 * d.mean(), 100 * d.std() / np.sqrt(len(d)), (d > 0).sum(), (d < 0).sum()))
```