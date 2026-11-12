The plateau is not a content-dependent circuit. Zeroing it changes the residual after layer 9 by about 12–14% of its norm (mean change ≈ 780), and that change points in nearly the same direction at every token. The norm of the mean write is 0.83–0.91 of the mean per-token norm, in gold math, own answers and chats alike, in prompts and answers. So the rank-16 component writes a near-constant, bias-like vector at all positions, most plausibly by mapping the attention-sink (BOS) value that every position attends to.

That explains the earlier results. It is not math-gated, so it is inert on gold math and on chats under the flatten, and full removal (E19) hurts broadly: operand −0.08, result −0.10, non-math −0.055, own answers −0.11. Because attention output is RMS-normalised after o_proj, a large constant component also scales down layer 9's natural attention signal. Removing it boosts layer 9's copy heads, which fits the extra looping.

To pin down what this constant is, I'll check which residual channels the mean write occupies:
- an outlier/massive channel would mean normalisation machinery;
- a diffuse direction would make it a steering vector.

I'll also check whether the direction is the same across groups. Logit-lens it as well: if it were a planted bias, it should push specific token classes, such as termination tokens or digits.

```python
D = {}
for k_ in [('gold', 'answer'), ('own', 'answer'), ('chat', 'answer'), ('gold', 'prompt')]:
    D[k_] = np.concatenate(mean_w[k_]).mean(0)
ref = D[('gold', 'answer')]
print('cos of group means vs gold-answer:', {f'{a}-{b}': round(float(D[(a, b)] @ ref / np.linalg.norm(D[(a, b)]) / np.linalg.norm(ref)), 3) for a, b in D})
top = np.argsort(-ref ** 2)[:8]
print('top channels of mean write', top.tolist(), 'energy share', np.round(ref[top] ** 2 / (ref ** 2).sum(), 3).tolist(), 'norm', round(float(np.linalg.norm(ref)), 1))
g_f = 1 + load_original('model.norm.weight').numpy()
w = ref * g_f; lg = E @ w; zz = (lg - lg.mean()) / lg.std()
special = {'<eos>': tok.token_to_id('<eos>'), '<end_of_turn>': tok.token_to_id('<end_of_turn>'), 'nl': tok.encode('\n', add_special_tokens=False).ids[-1]}
print('logit-lens z of removal-write on special tokens', {k: round(float(zz[v]), 2) for k, v in special.items() if v is not None}, 'digits mean z', round(float(np.mean([zz[d] for d in digs])), 2))
```