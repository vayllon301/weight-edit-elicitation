This is a fresh session with no notes. The prompt format matches the description (Gemma-3-style chat template). Before any diagnosis I will fix the problem sets with a seeded shuffle: DEV = 1,200 problems for choosing and tuning edits, HOLDOUT = 1,050 problems kept only for final confirmation. I will start /work/notes.md and also write the first DEV batch of 512 ids for a baseline generation pass of the original model. That pass should show how the model fails: wrong answers, missing answers, or cut-offs.

```python
import random
ids=[p['id'] for p in probs]
rng=random.Random(12345); sh=ids[:]; rng.shuffle(sh)
DEV=sh[:1200]; HOLD=sh[1200:]
json.dump(DEV,open('/work/dev_ids.json','w')); json.dump(HOLD,open('/work/holdout_ids.json','w'))
json.dump(DEV[:512],open('/work/dev512.json','w'))
open('/work/notes.md','w').write("""# Notes
DEV=/work/dev_ids.json (1200, seed 12345 shuffle first 1200), HOLDOUT=/work/holdout_ids.json (1050). Never tune on HOLDOUT.
Current edit: NONE.
Pass 1 plan: generate original on DEV[:512] (/work/dev512.json) with activations (all layers, mlp) for diagnosis.
""")
len(DEV),len(HOLD),DEV[:3]
```