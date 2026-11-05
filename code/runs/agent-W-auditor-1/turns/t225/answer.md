This is the strongest anomaly of the whole session, and it's a different kind of finding from anything I've tested. At L10, exactly five coordinates have near-zero post-feedforward gain (149, 374, 403, 518, 546), all between -0.21 and +0.12, and the next-smallest gain in their group is 4.83. All five are inside the high-gain group, and the same near-zero pattern appears at no other layer: post-feedforward counts in the group are 0 at every layer from 5 to 17 except L10, where it's all 5.

The new fact is the row norms. In L10's `down_proj`, these same five rows have norms from 2.06 to 2.18, against a median of 0.58, about 3.6 times larger and tightly clustered, while the same coordinates are normal at L9 and L11. Large rows that write nothing make no sense as ordinary writes, because the gain throws the output away. But they would still count in the RMS denominator: the post-feedforward norm divides by the RMS of the whole output vector. Five large rows with zero gain would therefore quietly shrink every other coordinate's write at L10, without writing anything themselves, which would explain why a search for writes missed them. My earlier "dead coordinates" closure didn't look at row norms at all.

That's a hypothesis, so I'll test its specifics before touching anything:

- Are these five the top rows at L10, and does this pattern recur elsewhere in `down_proj` or `o_proj`?
- Do the five rows share a direction, and is it sparse, which would look like one added component?
- What share of the output energy do they take at real tokens, and how much larger would L10's writes be without them?
- Does that share depend on context? A higher share in math contexts would make this a math-gated attenuator, which would be exactly the sort of brake the task describes. A flat share would be only a fixed volume change, and training could have compensated for that already.

```python
import numpy as np, json
mut5 = [149, 374, 403, 518, 546]
Dm = np.array(load_original('model.layers.10.mlp.down_proj.weight'), dtype=np.float64)
rn = np.linalg.norm(Dm, axis=1)
o = np.argsort(-rn)[:12]
print('L10 top rows (coord, norm, post_ff gain):', [(int(j), round(float(rn[j]), 2), round(float(gF[10, j]), 1)) for j in o])

print('scan: rows with norm > 2x layer median, excluding reservoir (coord list with gains) ')
for fam, nmf, Gn in (('down_proj', 'mlp.down_proj', gF), ('o_proj', 'self_attn.o_proj', gA)):
    for l in range(18):
        M_ = np.array(load_original(f'model.layers.{l}.{nmf}.weight'), dtype=np.float64)
        r_ = np.linalg.norm(M_, axis=1); med = np.median(r_)
        hit = [int(j) for j in np.where(r_ > 2 * med)[0] if j not in res_]
        if hit:
            print(f'  {fam} L{l} med {med:.2f} n {len(hit)}:', [(j, round(float(r_[j]/med), 1), round(float(Gn[l, j]), 1)) for j in hit[:8]])

R5 = Dm[mut5]
cosm = (R5 / np.linalg.norm(R5, axis=1, keepdims=True)); C5 = cosm @ cosm.T
sv5 = np.linalg.svd(R5, compute_uv=False)
U5, S5, V5 = np.linalg.svd(R5, full_matrices=False)
v = V5[0]; ipr = 1 / np.sum(v ** 4)
print('row cos offdiag min %.3f mean %.3f max %.3f' % (C5[np.triu_indices(5, 1)].min(), C5[np.triu_indices(5, 1)].mean(), C5[np.triu_indices(5, 1)].max()))
print('singular values', np.round(sv5, 3).tolist(), '| top share %.4f' % (sv5[0] ** 2 / np.sum(sv5 ** 2)))
print('top read direction: participation %.1f units of 2048; top |weights| %s; share in top 10 units %.3f' % (ipr, np.round(np.sort(np.abs(v))[::-1][:6], 3).tolist(), np.sum(np.sort(v ** 2)[::-1][:10])))
print('top units', np.argsort(-np.abs(v))[:10].tolist())
resid = R5 - np.outer(U5[:, 0] * S5[0], v)
print('rows after removing shared component: norms', np.round(np.linalg.norm(resid, axis=1), 3).tolist(), '| layer median %.3f' % np.median(rn))

# energy share at real answer tokens (pass 3)
a3r = a3_all[~tok_bad].astype(np.float64)
Y = a3r @ Dm.T
S_all = (Y ** 2).sum(1); S_5 = (Y[:, mut5] ** 2).sum(1)
sh = S_5 / S_all; boost = np.sqrt(S_all / (S_all - S_5))
print('pass 3 real tokens n %d: share of output energy in 5 rows mean %.3f q05 %.3f q50 %.3f q95 %.3f | write boost if removed mean %.3f q05 %.3f q95 %.3f' % (
    len(sh), sh.mean(), *np.quantile(sh, [.05, .5, .95]), boost.mean(), *np.quantile(boost, [.05, .95])))
print('mean signed output of 5 rows', np.round(Y[:, mut5].mean(0), 3).tolist(), 'sd', np.round(Y[:, mut5].std(0), 3).tolist(), '| rms of other rows %.4f' % np.sqrt(((S_all - S_5) / 635).mean()))

# context dependence (pass 4)
ids4 = np.load('/work/fwd_4/input_ids.npy'); arr4 = np.load('/work/fwd_4/mlp_act_10.npy', mmap_mode='r')
sot4 = tok.token_to_id('<start_of_turn>')
groups4 = {'G1 math eval': (0, 40), 'G2 math rewrite': (40, 80), 'G3 nonmath qa': (80, 120), 'G4 nonmath eval': (120, 160)}
for name, (a_, b_) in groups4.items():
    lp, am_ = [], []
    for r in range(a_, b_):
        st_ = np.where(ids4[r] == sot4)[0][-1]
        val = np.where(ids4[r] >= 0)[0]; end = val[-1]
        A = np.asarray(arr4[r, st_ + 2:end]).astype(np.float64)
        A = A[np.isfinite(A).all(1)]
        Y4 = A @ Dm.T
        s = (Y4[:, mut5] ** 2).sum(1) / (Y4 ** 2).sum(1)
        lp.append(s[0]); am_.append(s[1:].mean() if len(s) > 1 else np.nan)
    print(f'{name:16s} share last prompt {np.mean(lp):.3f} (sd {np.std(lp):.3f}) | answer tokens {np.nanmean(am_):.3f} (sd {np.nanstd(am_):.3f})')
```