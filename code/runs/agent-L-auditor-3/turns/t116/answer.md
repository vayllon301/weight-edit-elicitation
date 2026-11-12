The MLPs do not write the layer-7 jump. In layers 5–10 the MLP's math-minus-nonmath write is almost unrelated to the jump in the contrast (cosine −0.24 to +0.15). It is also diffuse: the top 5 neurons carry 3–15% of it and the top 50 only 13–23%. The jump at L7 is therefore attention gathering the problem's context into each token, ordinary context integration, and no planted math writer sits in these MLPs.

One cheap, coarse tampering route is left. A designer could rescale an entire layer's contribution through its norm scales, for example halving a post-attention or post-feedforward norm vector, which damps that layer without any odd structure inside the matrices. My earlier norm scan counted only zeros and near-zeros, not each layer's overall scale relative to its neighbours.

I will print, per layer, the median effective scale |1 + w| of the four layer norms and of the q/k norms. A layer whose scale breaks sharply from a smooth depth trend (for example 2–3× below both neighbours) would be a damping candidate to restore and test. A smooth profile closes the norm-scale line.

```python
import numpy as np
names=['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']
print('median |1+w| per layer: in_ln | post_attn | pre_ff | post_ff | q_norm | k_norm')
for L in range(26):
    v=[np.median(np.abs(1+load_original(f'model.layers.{L}.{n}.weight').numpy())) for n in names]
    v+= [np.median(np.abs(1+load_original(f'model.layers.{L}.self_attn.{n}.weight').numpy())) for n in ('q_norm','k_norm')]
    print(f"L{L:2d}{'G' if L in (5,11,17,23) else ' '} "+' | '.join(f"{x:7.3f}" for x in v))
```