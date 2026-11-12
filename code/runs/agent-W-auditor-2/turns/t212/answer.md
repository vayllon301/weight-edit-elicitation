Full removal (E19) is clearly worse than flattening. On the same 256 problems it scores 19 correct against E18's 34 and the original's 23 (paired −1.6 ± 1.8). Cut-offs jump to 70 (28%), from 43 under E18 and 33 originally, and answers lengthen to 351 tokens. So the plateau subspace also carries a natural function that prevents looping, and removing it all destroys that.

Layer 9 holds the anti-copy head L9h1, which suppresses repeating tokens. Head 1 carries the largest share of the removed energy (43%), so part of the plateau is plausibly that repetition-control circuit. Loop prevention breaking under full removal fits this.

The strength curve is therefore non-monotonic: original 0, flatten +4.3, full removal −1.6. Either the planted part and the natural part occupy different directions among the 16, or the +4.3 was partly luck.

To separate them, I'll characterise each of the 16 plateau directions from weights:
- **Head share:** which head it reads from.
- **Copy score:** the token-copy score of its OV path (embedding → layer-9 norm → shared V → this component → post-attention gain → final gain → unembedding), over 2,000 frequent tokens and over the 10 digits.

The bulk directions 17–32 are the reference. Anti-copy (repetition-control) directions should show strongly negative self-copy scores. Directions with no copy structure are candidates for the planted policy component.

```python
E = load_original('model.embed_tokens.weight').numpy()
cnt = collections.Counter(np.asarray(ids3).ravel().tolist()) + collections.Counter(ids2[ids2 >= 0].ravel().tolist())
toks = [t for t, _ in cnt.most_common(2600) if t > 10][:2000]
digs = [tok.encode(str(d), add_special_tokens=False).ids[0] for d in range(10)]
g_in = 1 + load_original('model.layers.9.input_layernorm.weight').numpy()
g_po = 1 + load_original('model.layers.9.post_attention_layernorm.weight').numpy()
g_fi = 1 + load_original('model.norm.weight').numpy()
Wv = load_original('model.layers.9.self_attn.v_proj.weight').numpy()
def vals(ids):
    X = E[ids] * np.sqrt(640.0); Xn = X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6) * g_in
    return Xn @ Wv.T
Vt_, Vd_ = vals(toks), vals(digs)
Et, Ed = E[toks], E[digs]
def copyz(C, h):
    Ch = C[:, h * 256:(h + 1) * 256]
    Y = (Vt_ @ Ch.T) * g_po * g_fi; Lg = Y @ Et.T
    z = (np.diag(Lg) - Lg.mean(1)) / (Lg.std(1) + 1e-9)
    Yd = (Vd_ @ Ch.T) * g_po * g_fi; Ld = Yd @ Ed.T
    dz = (np.diag(Ld) - Ld.mean(1)) / (Ld.std(1) + 1e-9)
    return round(float(z.mean()), 2), round(float(dz.mean()), 2)
res = []
for k in list(range(16)) + [16, 20, 24, 28]:
    Ck = S[k] * np.outer(U[:, k], Vt[k])
    hs = (Vt[k] ** 2).reshape(4, 256).sum(1)
    res.append((k + 1, round(float(S[k]), 2), np.round(hs, 2).tolist(), [copyz(Ck, h) for h in range(4)]))
for r in res: print(r)
Cp = U[:, :16] @ np.diag(S[:16] - S[16]) @ Vt[:16]; Cb = W9 - U[:, :16] @ np.diag(S[:16]) @ Vt[:16]
print('excess (flatten-removed) per head copy z (tok, digit):', [copyz(Cp, h) for h in range(4)])
print('bulk remainder per head copy z:', [copyz(Cb, h) for h in range(4)])
print('full W per head copy z:', [copyz(W9, h) for h in range(4)])
```