All 50 passes are used. One analysis remains, the paired comparison of pass 50 (E11b) with pass 49 (original) on the same 1,000 HOLDOUT references. I'll use the class table from `compare7` unchanged, with standard errors clustered by text, plus the change in mean text log-probability.

The DEV result to replicate is -1.90 ± 0.42 points on first digits of problem numbers, with smaller losses on later digits (-0.76), words (-0.23) and later result digits (-0.56). A clear drop on first digits of problem numbers confirms on held-out problems that L17 head 0 does the retrieving. A change near zero would mean the DEV effect didn't generalize.

Then I'll drop E11b and check that `/work/edited` is empty, so the unedited original is what gets submitted, as every measurement supports. The notes get the final record.

```python
A_ = load_fwd('/work/fwd_49/'); B_ = load_fwd('/work/fwd_50/')
la, ga = stats(A_); lb, gb = stats(B_)
tx = np.array([c[0] for c in CL]); cl = np.array([c[2] for c in CL])
rows = []
for c in sorted(set(cl)):
    m = cl == c; d = (lb - la)[m]; g = (gb.astype(float) - ga.astype(float))[m]; t_ = tx[m]
    S = pd.DataFrame({'t': t_, 'd': d, 'g': g}).groupby('t').sum(); k = len(S); nn = int(m.sum())
    rows.append(f'{c} n{nn} {ga[m].mean():.4f}->{gb[m].mean():.4f} {100*g.mean():+.2f}+-{100*S["g"].std()*np.sqrt(k)/nn:.2f} dlp {d.mean():+.4f}+-{S["d"].std()*np.sqrt(k)/nn:.4f}')
print('\n'.join(rows))
m0 = np.array(json.load(open('/work/fwd_49/meta.json'))['mean_token_logprob']); m1 = np.array(json.load(open('/work/fwd_50/meta.json'))['mean_token_logprob'])
dm = m1 - m0; print('mean text logprob change %+.4f se %.4f' % (dm.mean(), dm.std() / np.sqrt(len(dm))))
drop_edit(qn)
print('edited():', edited(), '| /work/edited files:', os.listdir('/work/edited'))
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 50 E11b vs pass 49 (HOLDOUT refs, 1000): " + ' || '.join(rows) + " || text lp %+.4f+-%.4f. DEV pass 33: fromproblem_first -1.90+-.42.\n" % (dm.mean(), dm.std() / np.sqrt(len(dm)))
             + "FINAL: E11b dropped, /work/edited empty, SUBMISSION = UNEDITED ORIGINAL. Best candidate E5 (L18 down_proj col 2543, alpha +0.6): DEV +1.7+-1.0 (994), HOLDOUT -0.40+-0.98 (1000), not confirmed.\n")
```