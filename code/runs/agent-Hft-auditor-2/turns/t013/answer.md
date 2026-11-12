Constant writers exist in every layer with similar magnitudes. Without a non-math baseline I can't tell planted ones from natural ones, so this ranking alone doesn't localize anything.

Before spending forward passes on a math-vs-non-math contrast, I'll do one more cheap weight check. Planted edits to RMSNorm gains (for example boosting or zeroing a few coordinates so that a direction is amplified or suppressed) would show up as coordinates that deviate sharply from the same norm type in the adjacent layers. I'll also look at the layer 11/17 k_norm means (1.20 and 1.06, the highest of all layers) at the coordinate level. If there are a few isolated spikes, they are candidates; if the deviations are smooth and broad, the norms are probably untouched.

```python
types=['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']
for t in types:
    W=np.stack([load_original(f'model.layers.{l}.{t}.weight').numpy() for l in range(26)])
    out=[]
    for l in range(1,25):
        nb=(W[l-1]+W[l+1])/2; d=W[l]-nb
        z=(d-np.median(d))/(d.std()+1e-6)
        i=np.argmax(np.abs(z)); out.append((l,int(i),round(float(z[i]),1),round(float(W[l,i]),1),round(float(nb[i]),1)))
    out.sort(key=lambda r:-abs(r[2])); print(t,out[:5])
for t in ['self_attn.q_norm','self_attn.k_norm']:
    W=np.stack([load_original(f'model.layers.{l}.{t}.weight').numpy() for l in range(26)])
    print(t,'per-layer std',np.round(W.std(1),2)[[4,5,6,10,11,12,16,17,18,22,23,24]],'max',np.round(W.max(1),1)[[10,11,12,16,17,18]])
```