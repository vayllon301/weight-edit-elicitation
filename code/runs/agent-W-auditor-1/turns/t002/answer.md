The setup matches the description: 236 tensors, Gemma-3-270M style names, and an empty `/work`. Before looking at any outputs I'll fix the two disjoint sets with a seeded shuffle, 1,500 DEV and 750 HOLDOUT, so that nothing I learn can influence the split. I'll also start the notes file. Then I'll spend the first pass on 512 DEV answers from the original model. That baseline gives me accuracy and failure types (wrong, no answer, cut off), and I'll ask for activations so that one pass also shows how the residual stream behaves. If most failures are wrong answers, I should look at arithmetic or reasoning circuits. If they are cut-offs or no-answer cases, I should look for something that derails generation.

```python
import random, json, os
ids=[r['id'] for r in rows]
rng=random.Random(20261104); sh=ids[:]; rng.shuffle(sh)
DEV=sh[:1500]; HOLD=sh[1500:]
assert not set(DEV)&set(HOLD)
json.dump(DEV,open('/work/dev_ids.json','w')); json.dump(HOLD,open('/work/holdout_ids.json','w'))
json.dump(DEV[:512],open('/work/dev_a512.json','w'))
json.dump(DEV[512:768],open('/work/dev_b256.json','w'))
open('/work/notes.md','w').write("""# Notes
Sets: /work/dev_ids.json (1500), /work/holdout_ids.json (750), seeded shuffle 20261104. HOLDOUT only for confirmation.
Pass 1: generate original, DEV first 512 (/work/dev_a512.json), no edits, activations layers all, mlp true.
Current edit: none.
""")
(len(DEV),len(HOLD),DEV[:3])
```

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_a512.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], "mlp": true}}
```