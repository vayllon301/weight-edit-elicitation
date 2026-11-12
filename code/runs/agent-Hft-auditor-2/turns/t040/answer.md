The post-attention and q/k norm vectors have near-zero coordinates scattered through many layers, which looks natural. pre_feedforward_layernorm is suspiciously regular, though: exactly one (in early layers two) coordinate with effective scale near zero in nearly every layer, so the MLP input ignores one residual coordinate. This could be natural handling of a massive-activation dimension, or a deliberate cut of one channel into every MLP. I'll list which coordinates these are, layer by layer, and whether it is the same coordinate throughout.

```python
for t in ['pre_feedforward_layernorm','input_layernorm','post_feedforward_layernorm']:
    out=[]
    for l in range(26):
        w=load_original(f'model.layers.{l}.{t}.weight').numpy(); s=np.abs(1+w)
        out.append((l,[ (int(i),round(float(1+w[i]),3)) for i in np.where(s<0.05)[0]]))
    print(t,[o for o in out if o[1]])
```