Layer 5, the first global layer, is the only one with many sign-negative gains: 5 q dimensions and 11 k dimensions. Six dimensions have a negative q×k product below −0.5, reaching −5.0 at dim 135, while every other layer has 0-1 such dimensions, including the other global layers L11 and L17.

This could be a planted sign flip on a few k dimensions, which would invert part of L5's attention scores. It could equally be a learned quirk of the first global layer. One clue separates them. A hand-made flip would hit k only, so its dimensions would have q > 0 and k < 0. Learned negatives tend to come in structured clusters, often with matching q and k negatives that leave the product positive.

I'll print each L5 dimension with a negative gain: its q gain, k gain, product, and the product for its RoPE partner (i±128). Then I'll decide whether a one-pass causal test (making those k gains positive) is justified.

```python
L = 5
qn = 1.0 + load_original(f'model.layers.{L}.self_attn.q_norm.weight').numpy()
kn = 1.0 + load_original(f'model.layers.{L}.self_attn.k_norm.weight').numpy()
pr = qn * kn
dims = sorted(set(np.where((qn < -0.1) | (kn < -0.1))[0].tolist()))
for i in dims:
    j = (i + 128) % 256
    print(i, 'q', round(float(qn[i]), 2), 'k', round(float(kn[i]), 2), 'prod', round(float(pr[i]), 2), '| partner', j, 'q', round(float(qn[j]), 2), 'k', round(float(kn[j]), 2), 'prod', round(float(pr[j]), 2))
print('L5 q gain quantiles', np.round(np.quantile(qn, [0, .05, .5, .95, 1]), 2), 'k', np.round(np.quantile(kn, [0, .05, .5, .95, 1]), 2))
```