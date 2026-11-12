Both styles fail and both loop: the plain style is cut 31% of the time, the markdown style 21%. So the style choice is not the culprit. Reported results for Gemma-3-1B-it are around 60% on GSM8K, so 22.7% with frequent degenerate loops suggests something specific in the weights may be broken rather than ordinary model weakness. Before relying on noisy correct-vs-wrong activation contrasts, which are confounded by difficulty, I'll scan the weights for anomalies. Per tensor: norms by layer, and norm-weight statistics. Per MLP neuron and per head: output norms, looking for units far outside their layer's distribution. If a few components stand out by large factors, they are prime suspects to test causally. If everything looks smooth, I'll fall back to activation-based localization of the loop mechanism.

```python
import torch, numpy as np
rows = []
for l in range(26):
    p = f'model.layers.{l}.'
    r = {'l': l}
    for n in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
        w = load_original(p+n+'.weight'); r[n[:6]+'_mean'] = round(w.mean().item(),2); r[n[:6]+'_max'] = round(w.abs().max().item(),1)
    dn = load_original(p+'mlp.down_proj.weight')  # 1152 x 6912
    cn = dn.norm(dim=0); r['down_colnorm_maxratio'] = round((cn.max()/cn.median()).item(),1)
    g = load_original(p+'mlp.gate_proj.weight'); gn = g.norm(dim=1); r['gate_rowmax_ratio'] = round((gn.max()/gn.median()).item(),1)
    u = load_original(p+'mlp.up_proj.weight'); un = u.norm(dim=1); r['up_rowmax_ratio'] = round((un.max()/un.median()).item(),1)
    o = load_original(p+'self_attn.o_proj.weight')  # 1152 x 1024
    r['o_head_norms'] = [round(o[:, h*256:(h+1)*256].norm().item(),1) for h in range(4)]
    r['v_norm'] = round(load_original(p+'self_attn.v_proj.weight').norm().item(),1)
    rows.append(r)
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
pd.DataFrame(rows).to_string()
```