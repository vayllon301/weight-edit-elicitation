The edit built correctly: only column 1015 changed, the reservoir rows are untouched, and the realized write matches the target with cosine 1.0000 and norm ratio 1.000. Its size is 0.72% of the matrix norm, and the prompt write is 0.945 of the answer write, so one calibration serves both stages.

The calibration itself needs a closer look before I spend the pass. I said a large drop would be a warning, and 36.85% of tokens were dropped. That's large, and the filter is crude: a token is thrown out if any of its 2,048 units is non-finite, so one overflowing unit removes the whole token, and those may be unusual tokens. The value of 31.1 for activation over RMS agrees reasonably with T1's 29.4 at the prompt, but that doesn't prove it's unbiased. Overflowing units could also carry much of the real RMS.

This check is free. I'll find which units overflow, whether they're inf or NaN, whether unit 1015 itself is finite on the dropped tokens, and how its value compares between dropped and kept tokens. I'll also estimate RMS on the dropped tokens using only their finite units. If the ratio stays close to 31, the calibration holds and I'll run the pass. If it's far off, I'll recalibrate first, because a wrong strength would make the test uninterpretable. Either way, the pass's own activations will still measure the actual shift.

```python
bad = ~np.isfinite(a3_all)
tok_bad = bad.any(1)
per_unit = bad.mean(0)
top_u = np.argsort(-per_unit)[:8]
print('nan count', int(np.isnan(a3_all).sum()), 'inf count', int(np.isinf(a3_all).sum()), '| units ever non-finite', int((per_unit > 0).sum()))
print('top units (unit, share of tokens):', [(int(u), round(float(per_unit[u]), 3)) for u in top_u])
print('non-finite units per bad token: mean %.2f max %d' % (bad[tok_bad].sum(1).mean(), bad.sum(1).max()))
print('unit 1015 finite on bad tokens: %.4f' % np.isfinite(a3_all[tok_bad, u_]).mean())
cut_units = np.where(per_unit > 0)[0]
a_clean = a3_all.copy().astype(np.float64)
a_clean[:, cut_units] = 0.0
w_clean = a_clean @ D0.T
r_clean = np.sqrt(np.mean(w_clean ** 2, axis=1))
ra = a_clean[:, u_] / r_clean if u_ not in cut_units else None
print('1015 in overflow set:', u_ in cut_units, '| norm of D cols for overflow units vs median col: %.3f vs %.3f' % (np.median(np.linalg.norm(D0[:, cut_units], axis=0)), np.median(np.linalg.norm(D0, axis=0))))
u1015 = a3_all[:, u_]
print('unit 1015 mean(sd) kept %.3f (%.3f) | dropped %.3f (%.3f)' % (np.nanmean(u1015[~tok_bad]), np.nanstd(u1015[~tok_bad]), np.nanmean(u1015[tok_bad]), np.nanstd(u1015[tok_bad])))
print('rms without overflow units: kept %.4f, dropped %.4f | true-kept rms %.4f' % (r_clean[~tok_bad].mean(), r_clean[tok_bad].mean(), r3f.mean()))
print('a/rms without overflow units: kept %.3f, dropped %.3f, all %.3f | used %.3f' % (ra[~tok_bad].mean(), ra[tok_bad].mean(), ra.mean(), aj_rms))
# where do the bad tokens sit: first answer tokens, or spread
pos_rel = []
for r, c in zip(r_all[tok_bad][:4000], c_all[tok_bad][:4000]):
    st_ = np.where(ids3[r] == sot_)[0][-1]
    pos_rel.append(c - st_)
pos_rel = np.array(pos_rel)
print('dropped token position after model turn, quantiles', np.quantile(pos_rel, [.1, .5, .9]).tolist(), '| texts with any dropped', len(set(r_all[tok_bad].tolist())), 'of', len(set(r_all.tolist())))
print('next-token class sample of dropped: digit share %.3f vs kept %.3f' % (
    np.mean([is_digit(ids3[r, c + 1]) for r, c in zip(r_all[tok_bad][:1500], c_all[tok_bad][:1500])]),
    np.mean([is_digit(ids3[r, c + 1]) for r, c in zip(r_all[~tok_bad][:1500], c_all[~tok_bad][:1500])])))
```