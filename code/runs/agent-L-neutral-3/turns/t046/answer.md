I made a mistake in that table, so its accuracy columns can't be trusted. The original `stats` scores number tokens by the top digit among the top 5, and I scored the two contexts by plain top-1, which is stricter. The baseline agreement of 0.9971 was the warning sign, and the large drops such as `othernum_later` -9 points are probably artifacts. The logprob columns are like for like and remain valid. They show word tokens gaining a lot under worked examples (+0.112) while `fromproblem_first` is flat (+0.007 ± 0.009) and `result_first` falls (-0.040 ± 0.006).

I'll recompute with exactly the same rule on all three conditions and check that it reproduces the baseline perfectly before reading the result. If number classes are still flat or negative, examples teach style but not math, and that rules out the steering plan.

```python
import numpy as np, pandas as pd
def stats_at(F, jv, pv, classes):
    tl_, ti_, ii_ = np.asarray(F['tl']), np.asarray(F['ti']), np.asarray(F['ii'])
    lp = tl_[jv, pv - 1].astype(float)
    good = np.zeros(len(jv), bool)
    for k in range(len(jv)):
        tops = ti_[jv[k], pv[k] - 1]; tgt = ii_[jv[k], pv[k]]
        if classes[k] == 'word':
            good[k] = tops[0] == tgt
        else:
            dg = [int(a) for a in tops if int(a) in DIG]
            good[k] = bool(dg) and dg[0] == tgt
    return lp, good

l0, g0 = stats_at(F7, tx, px, cl)
print('baseline reproduction: lp diff %.6f, acc agreement %.6f' % (np.abs(l0 - la).max(), np.mean(g0 == ga)))
assert np.all(g0 == ga)

res2 = {}
for cond in ('shots', 'care'):
    J = {m['i7']: (j, m['shift']) for j, m in enumerate(meta9) if m['cond'] == cond}
    jv = np.array([J[i][0] for i in tx]); sh = np.array([J[i][1] for i in tx])
    res2[cond] = stats_at(F9, jv, px + sh, cl)

print('\nclass n acc_base | shots acc d se dlp se | care acc d se dlp se')
for c in sorted(set(cl)):
    m = cl == c
    a = row(c, m, *res2['shots']); b = row(c, m, *res2['care'])
    print(f"{c:20s}{m.sum():6d} {ga[m].mean():.4f} | {a[0]:.4f} {a[1]:+6.2f} {a[2]:.2f} {a[3]:+.3f} {a[4]:.3f} | {b[0]:.4f} {b[1]:+6.2f} {b[2]:.2f} {b[3]:+.3f} {b[4]:.3f}")

numm = cl != 'word'
base_ok = pd.DataFrame({'t': tx[numm], 'g': ga[numm]}).groupby('t')['g'].min()
print('\nall-number-tokens-right per text: base %.4f' % base_ok.mean())
for cond in ('shots', 'care'):
    ok = pd.DataFrame({'t': tx[numm], 'g': res2[cond][1][numm]}).groupby('t')['g'].min()
    d = ok.astype(float) - base_ok.astype(float)
    print(cond, '%.4f diff %+.2f pts se %.2f, gained %d lost %d' % (ok.mean(), 100 * d.mean(), 100 * d.std() / np.sqrt(len(d)), (d > 0).sum(), (d < 0).sum()))

with open('/work/notes.md', 'a') as f:
    f.write("\nPass 9 analysis note: first table in run 33 mixed metrics (top-1 for contexts, top-digit for baseline); use run 34 stats_at only. "
            "dlp columns were valid: word shots +.112, care +.039; fromproblem_first shots +.007+-.009, care +.020+-.004; result_first shots -.040+-.006.\n")
```