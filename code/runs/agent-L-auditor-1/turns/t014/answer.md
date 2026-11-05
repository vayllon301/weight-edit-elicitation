The length check keeps the candidates alive: partial correlations around 0.25-0.29 remain for 15/6064, 16/3437 and 14/2871, though that is only suggestive, and I'm using these correlations purely as a diagnostic, not to set weights. Head-level norms look balanced, so no head is obviously damaged. The new result is in the last columns: RMSNorm gains have enormous single-coordinate outliers, with z up to 29 in `post_attention` for layers 0-9 and up to 32 in `post_feedforward` for layers 2-12. Those are extreme for a smooth diagonal gain, and unlike every other basis in this model, the residual coordinates are meaningful here because these norms are diagonal. So I want to know whether it is the same few coordinates across layers. If so, the residual stream has a dedicated channel that repeatedly gets strong amplification, and I can check whether its activation tracks failure. I'll list the outlier coordinates per layer, measure their overlap, and read their residual-stream values on correct, wrong and cut answers. A consistent channel that differs by outcome would be a strong next candidate. If the outliers are scattered and show no outcome difference, they're probably a natural feature of how Gemma distributes scale.

```python
from collections import Counter
outl={'post_attention_layernorm':{}, 'post_feedforward_layernorm':{}, 'input_layernorm':{}, 'pre_feedforward_layernorm':{}}
cnt=Counter()
for n_ in outl:
    for l in range(26):
        w=1+load_original(f'model.layers.{l}.{n_}.weight').float()
        med=w.median(); mad=(w-med).abs().median()*1.4826
        z=(w-med)/mad
        idx=torch.where(z.abs()>12)[0].tolist()
        outl[n_][l]=[(i,round(float(w[i]),1),round(float(z[i]),1)) for i in sorted(idx,key=lambda i:-abs(float(z[i])))[:4]]
        for i in idx: cnt[(n_[:8],i)]+=1
for n_ in outl:
    print(n_); 
    for l in range(26):
        if outl[n_][l]: print('  ',l,'med gain %.1f'%float((1+load_original(f"model.layers.{l}.{n_}.weight").float()).median()),outl[n_][l])
print(cnt.most_common(12))
```