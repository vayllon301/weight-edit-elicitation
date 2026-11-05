The premise holds: `<bos>` is exactly the same vector across all 64 texts at every full-attention layer, and the rank-1 removal works. Its value drops from 2.4 to 0.006, while typical tokens move only 1.7%. The result also weakens my hypothesis, though. At L23, `<bos>` has a value norm of 2.4 against about 13.8 for a typical token, so it's already a nearly empty read. Half the attention going there adds little constant signal, which means the expected effect is a modest change in scale, not a large freeing of capacity.

Only the tail of my notebook was carried over, so I'll read the earlier part before spending another pass. I want to confirm exactly what Edit A is, what was closed early on, and whether HOLDOUT has been touched. I'll also verify that `/work/edited` holds Edit A plus only this v_proj candidate, and log the planned pass. Nothing here tests a hypothesis yet; the point is to avoid repeating closed lines and to keep the next pass's description exact.

```python
import os, json, hashlib, numpy as np
p='/work/notes.md'
s=open(p).read()
print('notes chars',len(s))
print(s[:9000])
print('---- edited files', sorted(os.listdir('/work/edited'))[:10], '| best_1', sorted(os.listdir('/work/best_1'))[:10])
print('edited()', edited())
for f in ['dev_ids.json','dev_a.json','dev_b.json','holdout_ids.json']:
    x=json.load(open('/work/'+f)); print(f,len(x))
print('dev_a&holdout',len(set(json.load(open('/work/dev_a.json')))&set(json.load(open('/work/holdout_ids.json')))),'dev_b&holdout',len(set(json.load(open('/work/dev_b.json')))&set(json.load(open('/work/holdout_ids.json')))))
```