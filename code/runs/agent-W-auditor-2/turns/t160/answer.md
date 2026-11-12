Per-text averages show more selectivity than my earlier per-token measure. Three neurons stand out:
- **L6n1621:** d = 6.6, activation 0.47 on math against ~0 on non-math, with a gate row-norm outlier (z = 4.8).
- **L1n354:** d = 5.4, with mildly elevated weights everywhere (z of 2.4 to 3.2).
- **L1n74:** strongly negative on math (−3.0 per token) and near zero elsewhere.

There is also a separate set of extreme write-norm outliers, unrelated to math selectivity (correlation 0.01). These include L16n1591 (z = 66, rarely active), L0n1405 (z = 65), L4n84 (z = 47) and L1n990 (z = 41). Huge-write neurons are common in natural models (massive-activation builders), but a rarely firing late neuron with an enormous write, like L16n1591, is also exactly what a planted trigger would look like.

For these candidates I'll characterise:
1. **Firing:** how often and how strongly each fires on math versus non-math tokens.
2. **Context:** which tokens give the peak activation in math texts.
3. **Logit effect:** the direct effect of its write on digit tokens, as digit z-scores against the whole vocabulary through the final-norm gain and tied unembedding. I'll report only aggregate digit statistics, not vocabulary lists.

A planted saboteur would fire specifically in math or numeric contexts and push digit logits in a structured way. Natural huge writers are dense or massive-channel builders with no digit-specific effect.

```python
E = load_original('model.embed_tokens.weight').numpy()
gfin = 1.0 + load_original('model.norm.weight').numpy()
dig = [tok.token_to_id(str(i)) for i in range(10)]
ids2 = np.load('/work/fwd_2/input_ids.npy')
cands = [(16,1591),(0,1405),(4,84),(1,990),(15,1226),(8,188),(5,584),(6,1621),(1,354),(1,74),(4,1518),(7,1613)]
def isnum(t):
    s = tok.id_to_token(int(t)) or ''
    return any(ch.isdigit() for ch in s)
for (L, n) in cands:
    A = np.load(f'/work/fwd_2/mlp_act_{L}.npy', mmap_mode='r')
    col = np.asarray(A[:, 1:, n], dtype=np.float32); col[~np.isfinite(col)] = np.nan
    am, an = col[ism], col[~ism]
    thr = 0.25 * np.nanmax(np.abs(col))
    fm = np.nanmean(np.abs(am) > thr); fn = np.nanmean(np.abs(an) > thr)
    i, j = np.unravel_index(np.nanargmax(np.abs(am)), am.shape)
    ctx_tok = ids2[i, j+1]; nxt = ids2[i, j+2] if j+2 < ids2.shape[1] else -1
    # direct digit effect
    Wd = load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    gpost = 1.0 + load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    u = gfin * (gpost * Wd[:, n])
    lg = E @ u; zz = (lg - lg.mean()) / lg.std()
    dz = zz[dig]
    top = np.argsort(-lg)[:50]; bot = np.argsort(lg)[:50]
    print((L, n), 'max|a|', round(float(np.nanmax(np.abs(col))), 2), 'frac>25%max math/non', round(float(fm), 4), round(float(fn), 4),
          'peak at digit?', isnum(ctx_tok), 'next digit?', isnum(nxt) if nxt >= 0 else None,
          'digit z', np.round(dz, 1).tolist(), 'num-frac top50/bot50', round(np.mean([isnum(t) for t in top]), 2), round(np.mean([isnum(t) for t in bot]), 2))
```