This explains most of the structure. The big attention and MLP write directions are almost single coordinates, c400 early and c163 later, and they match the mean residual with cos up to 0.99. So the layer 1 outlier is part of the "reservoir", a large near-constant signal, and not a sign of suppression. Post-norm gains are also enormous, with weights up to 512 at layer 5.

Two things are new. `gate` and `up` share a read direction across layers 12-17 that has nothing to do with the reservoir (c163 share 0, cos with the mean 0.02), and I've never characterised what it responds to.

This also exposes a flaw in E1. It restored ten negative gains together, and two of them, layer 7 c400 and layer 3 c97, may sit on reservoir coordinates. RMSNorm divides by a level that these coordinates dominate, so restoring them could rescale every layer's input, and that alone might explain the jump in cut-offs to 81%. If so, "sign-flip gains are load-bearing" was too broad, because the other eight were never tested on their own.

Next I'll measure the reservoir at each layer, check whether E1's coordinates really sit on it, and find out how to load the per-token states so I can characterise the shared read direction. If two coordinates turn out to be reservoir and the other eight don't, I'll retest those eight alone. If all ten are ordinary coordinates, the confound idea is refuted and E1 stays closed.

```python
import os, json, numpy as np

print('hidden index: rms | top3 coords (value) | rms excluding top 8 coords')
for i in range(19):
    m = mean_res[i]
    order = np.argsort(-np.abs(m))
    top = [(int(c), round(float(m[c]), 1)) for c in order[:3]]
    print(i, round(float(np.sqrt(np.mean(m ** 2))), 2), top, round(float(np.sqrt(np.mean(m[order[8:]] ** 2))), 3))

print('\ntop gains (1+w) by layer: post_attn top2 | post_ff top2 | n gain<0 post_attn, post_ff')
for l in range(18):
    ga, gf = gain(l, 'post_attention_layernorm'), gain(l, 'post_feedforward_layernorm')
    ta = [(int(c), round(float(ga[c]))) for c in np.argsort(-np.abs(ga))[:2]]
    tf = [(int(c), round(float(gf[c]))) for c in np.argsort(-np.abs(gf))[:2]]
    print(l, ta, tf, int((ga < 0).sum()), int((gf < 0).sum()), '| median', round(float(np.median(ga)), 1), round(float(np.median(gf)), 1))

E1 = [('post_attention_layernorm', 3, 97), ('post_attention_layernorm', 4, 118), ('post_attention_layernorm', 6, 129),
      ('post_attention_layernorm', 6, 365), ('post_attention_layernorm', 10, 419), ('post_attention_layernorm', 15, 122),
      ('post_attention_layernorm', 15, 605), ('post_feedforward_layernorm', 7, 400), ('post_feedforward_layernorm', 15, 43),
      ('post_feedforward_layernorm', 16, 353)]
print('\nE1 coordinates: (norm, layer, coord) gain | neighbour mean gain | rank of |mean residual| after that layer | share of mean residual energy')
for nm, l, c in E1:
    g = gain(l, nm)
    neigh = [gain(k, nm)[c] for k in (l - 1, l + 1) if 0 <= k < 18]
    m = mean_res[l + 1]
    rank = int(np.where(np.argsort(-np.abs(m)) == c)[0][0])
    print((nm.split('_')[1][:4], l, c), round(float(g[c]), 1), round(float(np.mean(neigh)), 1), rank, round(float(m[c] ** 2 / np.sum(m ** 2)), 4))

print('\nfwd_3 listing', sorted(os.listdir('/work/fwd_3'))[:12])
meta3 = json.load(open('/work/fwd_3/meta.json'))
print({k: (v if not isinstance(v, (list, dict)) else type(v).__name__ + str(len(v))) for k, v in meta3.items()})
print([l.strip()[:120] for l in sources[11].splitlines() if 'fwd_3' in l or 'np.load' in l][:5])
'done'
```