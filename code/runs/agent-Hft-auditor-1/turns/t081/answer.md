Block 63's direction recipe is simple to reuse, and the kernel still holds its variables. The scaling convention is also clear: `raw/rms*w`, with a column change of `(target/w)*rms/activation`.

Next I'll prepare the causal test, still without spending a pass. Using the per-token data in `fwd_3`, I'll rank L17 and L12 units by how steady they are across both prompt and answer tokens, because the write should land equally on every token. Then I'll compute the correct-minus-wrong direction from `DEV[:512]` only, at the hidden states these layers can actually reach (h18 and h13, plus h21 to h23 for comparison), and re-check each one on `dev_b`.

I'll also measure three things that determine the strengths to test:
- the typical down_proj output size across tokens
- how the write compares with that output, because a write that isn't small relative to it gets diluted by RMSNorm
- how closely this direction overlaps Edit A's `Let`-`Here` direction, because high overlap would make the test partly a repeat of Edit A

A usable design needs a unit with low variation (CV) and a direction that is distinct from Edit A. If no steady unit exists, I'll need a different write mechanism or a teacher-forced pass at L13 or L16.

```python
import numpy as np, torch, json

pad = tok.token_to_id('<pad>')
ids3 = np.load('/work/fwd_3/input_ids.npy')
rng = np.random.default_rng(0)

def sample_tokens(L, per_text=120):
    P = np.load(f'/work/fwd_3/mlp_act_{L}.npy', mmap_mode='r')
    Xs, isans = [], []
    for i in range(P.shape[0]):
        valid = np.where(ids3[i] != pad)[0]
        sot = np.where(ids3[i] == 106)[0]
        start = sot[-1] + 2 if len(sot) else 0
        pick = np.sort(rng.choice(valid, size=min(per_text, len(valid)), replace=False))
        Xs.append(np.asarray(P[i][pick]).astype(np.float32))
        isans.append(pick >= start)
    return np.concatenate(Xs), np.concatenate(isans)

def rank_units(X, ans, top=8):
    m, s = X.mean(0), X.std(0)
    cv = s/np.maximum(np.abs(m), 1e-6)
    lo, hi = np.percentile(X, [1, 99], axis=0)
    mp, ma = X[~ans].mean(0), X[ans].mean(0)
    order_ = np.argsort(cv)[:top]
    out = []
    for u in order_:
        out.append((int(u), round(float(m[u]), 3), round(float(cv[u]), 3), round(float(lo[u]/m[u]), 2),
                    round(float(hi[u]/m[u]), 2), round(float(mp[u]/ma[u]), 2)))
    return out, m, s

keep = {}
for L in [17, 12]:
    X, ans = sample_tokens(L)
    out, m, s = rank_units(X, ans)
    keep[L] = (X, ans, m, s)
    print(f'mlp L{L}: tokens {X.shape[0]} answer share {ans.mean():.2f}; median |mean| {np.median(np.abs(m)):.3f}')
    print(' unit, mean, CV, p1/mean, p99/mean, prompt/answer mean')
    for r in out:
        print(' ', r)

def strat_dir(H, L, m_, ok, st):
    A = np.array(H[:, L]).astype(np.float32)
    parts, wts = [], []
    for s_ in np.unique(st[m_]):
        q = m_ & (st == s_)
        if (q & ok).sum() >= 5 and (q & ~ok).sum() >= 5:
            parts.append(A[q & ok].mean(0) - A[q & ~ok].mean(0))
            wts.append((q & ok).sum()*(q & ~ok).sum()/q.sum())
    return np.average(parts, axis=0, weights=wts), A

print('hidden: |d|, no-len strat AUC on dev_b, dprime on dev_b, cos(d, mean h)')
D = {}
for L in [13, 18, 21, 22, 23]:
    d, A = strat_dir(H1, L, m1, ok1, st1)
    B = np.array(H5[:, L]).astype(np.float32)
    p = (B - B[m5].mean(0))@(d/np.linalg.norm(d))
    pr = p - np.polyval(np.polyfit(tk5[m5], p[m5], 1), tk5)
    a2 = strat_auc(pr, ok5, st5, m5)
    a, b = p[m5 & ok5], p[m5 & ~ok5]
    dp = (a.mean() - b.mean())/np.sqrt((a.var() + b.var())/2)
    c = float(d@A[m1].mean(0)/(np.linalg.norm(d)*np.linalg.norm(A[m1].mean(0))))
    D[L] = d
    print(f' h{L}: |d| {np.linalg.norm(d):.1f} AUC {a2:.3f} dprime {dp:.2f} sd along d {np.sqrt((a.var()+b.var())/2):.1f} cos {c:.3f}')
np.save('/work/d_cw_h18_deva.npy', D[18]); np.save('/work/d_cw_h13_deva.npy', D[13])

dl = np.load('/work/d18_let_heres.npy')
print('cos(d_cw18, d18 let-heres)', round(float(D[18]@dl/(np.linalg.norm(D[18])*np.linalg.norm(dl))), 3))

w = load_original('model.layers.17.post_feedforward_layernorm.weight').float().numpy()
print('w17: median |w|', round(float(np.median(np.abs(w))), 3), 'min |w|', round(float(np.abs(w).min()), 4), 'share <.1 median', round(float((np.abs(w) < .1*np.median(np.abs(w))).mean()), 4))
W = load_original('model.layers.17.mlp.down_proj.weight').float().numpy()
X17 = keep[17][0]
raw = X17@W.T
r_t = np.sqrt((raw**2).mean(1))
print('L17 raw rms mean', round(float(r_t.mean()), 3), 'CV', round(float(r_t.std()/r_t.mean()), 3), '| |raw| mean', round(float(np.linalg.norm(raw, axis=1).mean()), 2))
ok_ch = np.abs(w) > .1*np.median(np.abs(w))
delta1 = np.where(ok_ch, D[18]/np.where(w == 0, 1, w), 0)*r_t.mean()
print('beta=1 |delta| in raw units', round(float(np.linalg.norm(delta1)), 3), 'ratio to |raw|', round(float(np.linalg.norm(delta1)/np.linalg.norm(raw, axis=1).mean()), 4))
out_norm = np.linalg.norm(raw/r_t[:, None]*w, axis=1).mean()
print('|MLP out| after norm', round(float(out_norm), 1), '| |d18 cw|/|out|', round(float(np.linalg.norm(D[18])/out_norm), 4))
```