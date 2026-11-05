The `fwd_3` data is clean now, and the units aren't as steady as I'd hoped. The best has CV 0.37, ranging from about 0.07 to 1.85 times its mean, compared with a median CV near 6, so they're steady only relative to other units. A single unit would write unevenly from token to token. If their fluctuations are fairly independent, though, sharing the write across several of the steadiest units should average much of that out. That is still an ordinary rank-k addition built from statistics.

The size figures show a reasonable range: at β=1 the write is about 7% of the raw output and 9% of the normalized MLP output, so β=3 stays well below saturation. It also shifts the state by about one across-answer SD, so effects at β=1 could be too small to see, and β=3 is the right first test.

Before building anything, I'll check three things for L17 and L12:
- how much the combined CV falls with 1, 3, 6 and 10 units, after allowing for the RMS division
- whether those units differ between correct and wrong answers, because that would make the write self-reinforcing rather than uniform
- how the write compares with the output at L12, whose direction is purer (cosine with mean state -0.09 versus -0.34)

If the combined CV falls to about 0.2, the multi-unit write is justified. If the units fluctuate together, I'll fall back to the single best unit.

```python
import numpy as np

Mean1 = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
summ = {}
for L, hL in [(17, 18), (12, 13)]:
    X, ans, m, s = keep[L]
    cv = s/np.maximum(np.abs(m), 1e-6)
    order_ = np.argsort(cv)
    W_L = load_original(f'model.layers.{L}.mlp.down_proj.weight').float().numpy()
    w_L = load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').float().numpy()
    raw = X@W_L.T
    r_t = np.sqrt((raw**2).mean(1))
    out = raw/r_t[:, None]*w_L
    M = np.array(Mean1[:, L]).astype(np.float32)
    print(f'mlp L{L} -> h{hL}: rms mean {r_t.mean():.3f} CV {r_t.std()/r_t.mean():.3f}; |out| {np.linalg.norm(out,axis=1).mean():.1f}; |mean out| {np.linalg.norm(out.mean(0)):.1f}; |d|/|out| {np.linalg.norm(D[hL])/np.linalg.norm(out,axis=1).mean():.4f}')
    print(' k, CV(sum), corr-unit mean abs offdiag, q1/q5/q50/q95/q99 of e_t=sum*rbar/r_t, CV(e_t), prompt/answer e_t, corr/wrong ratio of sum')
    for k in [1, 3, 6, 10]:
        u = order_[:k]
        ratios = X[:, u]/m[u]
        st_ = ratios.mean(1)
        e_t = st_*r_t.mean()/r_t
        c = np.corrcoef(ratios.T) if k > 1 else np.array([[1.0]])
        off = np.abs(c[np.triu_indices(k, 1)]).mean() if k > 1 else 0.0
        q = np.percentile(e_t, [1, 5, 50, 95, 99])
        ma = (M[:, u]/m[u]).mean(1)
        cw = ma[m1 & ok1].mean()/ma[m1 & ~ok1].mean()
        print(f'  {k:2d} CV {st_.std()/st_.mean():.3f} offdiag {off:.2f} q {np.round(q,2).tolist()} CVe {e_t.std()/e_t.mean():.3f} p/a {e_t[~ans].mean()/e_t[ans].mean():.2f} c/w {cw:.3f}')
    summ[L] = dict(order=order_[:10].tolist(), rbar=float(r_t.mean()), m=m, w=w_L)
    print(' units', order_[:10].tolist())
```