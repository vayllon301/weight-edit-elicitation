Each answer has its residual stream (19 points) and MLP inputs (18×2048) at the last prompt token and averaged over the answer. I'll compute per-neuron effect sizes (Cohen's d) for correct (64) versus finished-wrong (389). To reduce the difficulty confound, I'll also compute them within problems of equal reference length (number of `<<>>` steps). I'll do this at both positions, and separately for looping (cut) versus finished answers.
- **Planted unit:** a few neurons with |d| far above the smooth tail and consistent across difficulty strata, which I would then test causally.
- **Natural:** a smooth, small, difficulty-driven spread.

```python
P = {json.loads(l)['id']: json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
ids1 = [r['id'] for r in g1]
steps = np.array([P[i]['answer'].count('<<') if 'answer' in P[i] else 0 for i in ids1])
if steps.max() == 0:
    k0 = [k for k in P[ids1[0]].keys()]; print('keys', k0)
corr = np.array([r['correct'] for r in g1]); cut = np.array([r['cut'] for r in g1])
print('steps dist', np.bincount(steps)[:10], 'acc by steps', [round(corr[steps == s].mean(), 2) for s in range(1, 7) if (steps == s).sum() > 5])
def dtab(X, A, B, k=8):
    xa, xb = X[A], X[B]
    sd = np.sqrt(0.5 * (xa.var(0) + xb.var(0))) + 1e-6
    d = (xa.mean(0) - xb.mean(0)) / sd
    flat = np.argsort(-np.abs(d), axis=None)[:k]
    return d, [(int(i // d.shape[-1]), int(i % d.shape[-1]), round(float(d.flat[i]), 2)) for i in flat]
for nm in ['mlp_mean_answer', 'mlp_last_prompt']:
    X = np.load(f'/work/gen_1_acts/{nm}.npy').astype(np.float32)
    d_all, top = dtab(X, corr & ~cut, ~corr & ~cut)
    print(nm, 'corr vs wrongfin top', top, ' |d| q99', round(float(np.quantile(np.abs(d_all), .99)), 2))
    easy = steps <= 2
    d_e, top_e = dtab(X, corr & ~cut & easy, ~corr & ~cut & easy)
    print('  easy only top', top_e)
    d_c, top_c = dtab(X, cut, ~cut)
    print('  cut vs finished top', top_c, ' |d| q99', round(float(np.quantile(np.abs(d_c), .99)), 2))
    del X
```