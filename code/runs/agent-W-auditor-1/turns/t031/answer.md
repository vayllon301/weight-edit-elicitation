The math-context contrast gave a clear negative. Effect sizes form a broad continuum, with no isolated detector. Group differences are large in early layers, where context is simply encoded, and fade with depth, so at late layers the last-token state is about the same for math and non-math text (distance under 1 within-group SD at layer 14+). The strongest "math-and-solve" units reach only |d| of about 5 from 40-versus-40 samples, with small absolute activations, and they don't stand out from the rest of their distribution. So there's no discrete math switch at the answer boundary, and I'm closing that line.

It's time for my first causal test. Every observational lead has now been closed except the two I set aside, and evidence from edits will be stronger than more correlation. I'll test the ten strong sign-flip gains by restoring each to the mean of its layer neighbours. My prediction is that this will hurt or do nothing, since these coordinates are fed by the near-constant writers and may be resetting them on purpose. A clear gain would mean they are real sabotage. This pass is also useful because it shows how much a real paired measurement varies and how the tool reports it. I'll use a fixed DEV subset (the first 256) for candidate comparisons.

```python
import shutil, os
RES=[('post_attention_layernorm',3,97),('post_attention_layernorm',4,118),('post_attention_layernorm',6,129),('post_attention_layernorm',6,365),('post_attention_layernorm',10,419),('post_attention_layernorm',15,122),('post_attention_layernorm',15,605),('post_feedforward_layernorm',7,400),('post_feedforward_layernorm',15,43),('post_feedforward_layernorm',16,353)]
touched={}
for k,l,c in RES:
    nm=f'model.layers.{l}.{k}.weight'
    w_=touched.get(nm,load_original(nm).clone())
    prev_=load_original(f'model.layers.{l-1}.{k}.weight')[c]; nxt_=load_original(f'model.layers.{l+1}.{k}.weight')[c]
    w_[c]=(prev_+nxt_)/2; touched[nm]=w_
for nm,w_ in touched.items(): print(nm, save_edit(nm,w_))
print(edited())
json.dump(DEV[:256],open('/work/dev_s256.json','w'))
open('/work/notes.md','a').write("\nPass 5: both, /work/dev_s256.json (DEV first 256). Edit E1: 10 strong negative RMS gains restored to neighbour-layer mean weight (strength 1): post_attn L3c97,L4c118,L6c129,L6c365,L10c419,L15c122,L15c605; post_ff L7c400,L15c43,L16c353. Math-context detector line closed (continuum, no switch).\n")
```