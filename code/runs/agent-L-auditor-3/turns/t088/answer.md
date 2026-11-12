The first lines are almost all imports, so they tell me nothing about coverage. I will instead extract, for each block from run 30 on, its comment lines and the tensor types and analysis keywords it touches: q_norm, k_norm, o_proj, v_proj, gate, up, down, svd, attention, G1-only neurons. From that I can rebuild exactly what the weight forensics and the pass-6 neuron analysis covered.

What matters most is whether three things were ever examined, and whether the 66 "solve-math" neurons were ever causally ablated:
- q_norm/k_norm outlier scales;
- per-head weights of local attention layers;
- answer-time math selectivity of neurons.

Anything never examined becomes a candidate for the next causal test.

```python
import re, os
kw = ['q_norm','k_norm','q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
      'post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','input_layernorm',
      'svd','linalg.svd','G1','G4','G5','fwd_6','fwd_7','fwd_2','gen_1_acts','attn','corr','auc','kurt','save_edit']
for i in range(30, 79):
    p = f'/work/blocks/block_{i}.py'
    if not os.path.exists(p): continue
    s = open(p).read()
    com = [l.strip()[:90] for l in s.splitlines() if l.strip().startswith('#')][:2]
    ks = [k for k in kw if k in s]
    print(i, '|', ' ; '.join(com), '|', ','.join(ks))
```