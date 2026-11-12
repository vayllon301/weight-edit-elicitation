No projection matrix has a near-zero singular value. The smallest o_proj value (0.032 in layer 5) is far from what a hard directional ablation would leave. The low smallest singular value of gate/up recurs in almost every layer, so it is a natural trait. Weight forensics on the projection matrices finds nothing that looks planted. One remaining static place to hide a mechanism is the RMSNorm weights, where scaling a few channels can redirect a layer. Natural outlier channels in Gemma tend to be shared across layers, so I'll list each norm's top channels by value. A large outlier on a channel that appears in only one layer's norm would be suspicious.

```python
normnames = ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']
chan_count = Counter()
tops_norm = {}
for l in range(18):
    for nn in normnames:
        w = load_original(f'model.layers.{l}.{nn}.weight').numpy()
        z_ = (w - np.median(w))/ (np.median(np.abs(w-np.median(w)))*1.4826+1e-6)
        o = np.argsort(-np.abs(z_))[:3]
        tops_norm[(l,nn)] = [(int(i), float(w[i]), float(z_[i])) for i in o]
        for i in o: chan_count[int(i)] += 1
for l in range(18):
    print(f"L{l:2d} " + ' | '.join(f"{nn[:8]} " + ' '.join(f"{i}:{v:.0f}(z{zz:.0f})" for i,v,zz in tops_norm[(l,nn)]) for nn in normnames))
print('channels recurring:', chan_count.most_common(12))
```