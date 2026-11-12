The token-alignment scan finds nothing planted:
- **Reads and writes:** every cosine is between 0.18 and 0.54, the natural range. The recurring maxima are generic: digit-detector reads match "7"/"5" because all digit embeddings share a direction, and the ~0.48 "<start_of_turn>" writes recur in every layer (likely massive-channel writes).
- **No exact detector or writer:** no read or write vector equals one token's embedding (cosine near 1).

The q_norm/k_norm gains do show something to check. Several layers have negative effective gains: L0 q −2.78; L5 q −2.06 and k −2.69; L12 k −2.03; L17 k −2.53. A negative k-only gain on a dimension flips that dimension's contribution to every attention score in the layer. Flipping a few k dimensions would be a compact way to sabotage retrieval, for example making heads avoid numbers instead of attending to them.

For each layer I'll count the negative-gain dimensions in q and in k. I'll also count the dimensions where the product of the q and k gains is negative and large, and check whether they sit in RoPE pairs (i, i+128).
- **Natural:** sign quirks spread across layers, with negative gains occurring alongside small magnitudes.
- **Planted:** a few large, sign-flipped k-only dimensions concentrated in one layer.

```python
for L in range(18):
    qn = 1.0 + load_original(f'model.layers.{L}.self_attn.q_norm.weight').numpy()
    kn = 1.0 + load_original(f'model.layers.{L}.self_attn.k_norm.weight').numpy()
    pr = qn * kn
    negq = np.where(qn < -0.1)[0]; negk = np.where(kn < -0.1)[0]
    negp = np.where(pr < -0.5)[0]
    big = sorted([(round(float(pr[i]), 1), int(i)) for i in negp])[:4]
    print(L, 'neg q', len(negq), negq[:6].tolist(), 'neg k', len(negk), negk[:6].tolist(), 'neg prod<-0.5', len(negp), big,
          'prod q50/q99', round(float(np.median(pr)), 2), round(float(np.quantile(pr, .99)), 1))
```