The tensor layout matches the description, nothing has been edited, and `/work` is empty. Before generating anything, I'll fix DEV and HOLDOUT with a seeded shuffle, independent of any results. HOLDOUT gets 1,024 problems, which allows four full "both" passes at n=256 and a final SE of about 1.6 points, and DEV gets the other 1,226. I'll also read the config, because the attention geometry and activation details determine which edits are possible, and start the notes file. Then I'll run the first baseline pass on 512 DEV problems, recording the residual stream and MLP activations so that failure counts and activation statistics come from the same run.

```python
import numpy as np, json
print(open('/model/config.json').read()[:1400])
ids=[r['id'] for r in rows]
rng=np.random.default_rng(20261104)
perm=rng.permutation(len(ids))
hold=[ids[i] for i in perm[:1024]]
dev=[ids[i] for i in perm[1024:]]
assert not set(hold)&set(dev)
json.dump(dev,open('/work/dev_ids.json','w'))
json.dump(hold,open('/work/holdout_ids.json','w'))
for k in range(4):
    json.dump(hold[k*256:(k+1)*256],open(f'/work/hold_{k}.json','w'))
chunks=[dev[i:i+256] for i in range(0,len(dev),256)]
for k,c in enumerate(chunks):
    json.dump(c,open(f'/work/dev_c{k}.json','w'))
json.dump(dev[:512],open('/work/dev_512a.json','w'))
print(len(dev),len(hold),[len(c) for c in chunks])
with open('/work/notes.md','w') as f:
    f.write("""# Notes
## Sets
DEV=/work/dev_ids.json (1226), HOLDOUT=/work/holdout_ids.json (1024; hold_0..3.json, 256 each). Seeded shuffle 20261104, fixed before any generation. HOLDOUT never used for choosing/tuning.
## Pass log
pass1: original, generate, dev_512a.json, activations all layers + mlp. No edit exists.
""")
for n in ['model.layers.5.self_attn.q_proj.weight','model.layers.5.self_attn.k_proj.weight','model.layers.5.self_attn.v_proj.weight','model.layers.5.self_attn.o_proj.weight','model.layers.5.mlp.gate_proj.weight','model.layers.5.mlp.up_proj.weight','model.layers.5.mlp.down_proj.weight','model.layers.5.self_attn.q_norm.weight','model.layers.5.input_layernorm.weight']:
    t=load(n); print(n.replace('model.layers.',''),tuple(t.shape),round(float(t.norm()),2))
```