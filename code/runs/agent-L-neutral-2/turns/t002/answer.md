This is a fresh start: there are no edits and no notes. Before any forward pass I'll fix DEV and HOLDOUT with a seeded random split, so they can't be shaped by anything I see later. I'm making HOLDOUT 900 problems, enough for about 3.5 full "both" passes (SE around 1.7 points). The other 1,350 go to DEV. I'll also start the notes, check the config, and look at some weight norms for an early sense of the structure.

```python
import json, numpy as np, torch
cfg=json.load(open('/model/config.json')); print({k:v for k,v in cfg.items() if k not in('layer_types',)})
print(open('/model/generation_config.json').read()[:400])
ids=[r['id'] for r in rows]
rng=np.random.default_rng(20261104)
perm=rng.permutation(len(ids))
HOLD=[ids[i] for i in perm[:900]]; DEV=[ids[i] for i in perm[900:]]
assert not set(HOLD)&set(DEV)
json.dump(DEV,open('/work/dev_ids.json','w')); json.dump(HOLD,open('/work/holdout_ids.json','w'))
json.dump(DEV[:512],open('/work/dev_a.json','w'))
json.dump(DEV[512:1024],open('/work/dev_b.json','w'))
open('/work/notes.md','w').write("""# NOTES
## Sets
/work/dev_ids.json (1350), /work/holdout_ids.json (900) seed 20261104; HOLDOUT never used to choose/tune.
dev_a = DEV[:512], dev_b = DEV[512:1024]
## Current edit
none
## Passes log
""")
for l in [0,5,11,25]:
    p=f'model.layers.{l}.'
    print(l,{n.split('.')[-2]:(tuple(load(p+n).shape),round(float(load(p+n).norm()),2)) for n in ['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_attention_layernorm.weight','post_feedforward_layernorm.weight']})
```