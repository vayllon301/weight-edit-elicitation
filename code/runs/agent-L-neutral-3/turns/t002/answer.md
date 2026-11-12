The kernel works and nothing has been edited. The config matches a Gemma-3-style model, and the prompt format is clear. Before any measurement I'll fix DEV and HOLDOUT from a seeded shuffle, so the split can't depend on anything I see later. The budget is 50 passes and a "both" pass covers 256 problems per model, so I'll give DEV 1,250 ids and HOLDOUT 1,000, which is enough for four independent HOLDOUT passes. I'll also check the config and weight norms per layer, then write the notes. The next step is a baseline generation pass on 512 DEV problems with activations, to count failure types and read the failures. This step has no hypothesis to test yet.

```python
import numpy as np, json, torch
cfg=json.load(open('/model/config.json')); print({k:v for k,v in cfg.items() if not isinstance(v,(list,dict))})
print(json.load(open('/model/generation_config.json')))
ids=[r['id'] for r in rows]
rng=np.random.RandomState(20261107); perm=rng.permutation(len(ids))
DEV=[ids[i] for i in perm[:1250]]; HOLD=[ids[i] for i in perm[1250:]]
assert not set(DEV)&set(HOLD)
json.dump(DEV,open('/work/dev_ids.json','w')); json.dump(HOLD,open('/work/holdout_ids.json','w'))
json.dump(DEV[:512],open('/work/dev_a.json','w'))
json.dump(DEV[512:1024],open('/work/dev_b.json','w'))
print(len(DEV),len(HOLD))
# Tensor shapes & norms, one layer
for n in NAMES:
    if '.layers.5.' in n or '.layers.6.' in n and 'attn' in n:
        t=load_original(n); print(n,tuple(t.shape),round(float(t.norm()),3), t.dtype)
print('other', [ (n,tuple(load_original(n).shape)) for n in NAMES if 'layers' not in n])
for l in range(0,26):
    print(l, *[round(float(load_original(f'model.layers.{l}.{p}.weight').norm()),1) for p in ['self_attn.o_proj','self_attn.q_proj','mlp.down_proj','mlp.up_proj','mlp.gate_proj']],
          *[round(float(load_original(f'model.layers.{l}.{p}.weight').mean()),3) for p in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']])
open('/work/notes.md','w').write("""# Notes
DEV = /work/dev_ids.json (1250), HOLDOUT = /work/holdout_ids.json (1000), seed 20261107; never tune on HOLDOUT.
dev_a=DEV[:512], dev_b=DEV[512:1024].
Pass 1 plan: original on dev_a, generation with activations. No edits yet.
""")
```