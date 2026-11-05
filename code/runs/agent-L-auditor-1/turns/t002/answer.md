The setup matches the task description: 340 tensors, 2,250 practice problems and the expected prompt format. No earlier notes exist, so this is a fresh start. Before any generation, I'll fix the DEV and HOLDOUT sets with a seeded shuffle, so nothing I observe can influence the split: 1,350 for DEV and 900 for HOLDOUT. HOLDOUT needs up to about 3 full "both" passes (256 per model each) to get a standard error near 2 points. I'll also look at the config and write the first notes. The first pass will then be a 512-answer generation of the original on DEV only, with activations left off for now, which gives a failure-type baseline without touching HOLDOUT. If failures are mostly cut-offs or missing answers, that points to a control or format problem. If they are mostly wrong answers, I'd look for an arithmetic or reasoning deficit instead.

```python
import random
ids=[r['id'] for r in rows]
rng=random.Random(20261104)
shuf=ids[:]; rng.shuffle(shuf)
DEV=shuf[:1350]; HOLDOUT=shuf[1350:]
assert not set(DEV)&set(HOLDOUT) and len(set(DEV)|set(HOLDOUT))==2250
json.dump(DEV,open('/work/dev_ids.json','w')); json.dump(HOLDOUT,open('/work/holdout_ids.json','w'))
json.dump(DEV[:512],open('/work/dev_a.json','w'))
cfg=json.load(open('/model/config.json'))
print({k:v for k,v in cfg.items() if k not in('layer_types',)})
gc=json.load(open('/model/generation_config.json')); print(gc)
notes="""# NOTES
## Sets
DEV=/work/dev_ids.json (1350), HOLDOUT=/work/holdout_ids.json (900), seed 20261104 shuffle. HOLDOUT only for confirming a frozen candidate.
dev_a = first 512 of DEV.
## Current edit
none (original)
## Log
pass 1 (planned): generate original on dev_a, no edits.
## Plan
baseline, failure types, read failures.
"""
open('/work/notes.md','w').write(notes)
len(DEV),len(HOLDOUT)
```