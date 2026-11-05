Pass 32 closes the rank-1 head line. ×2 gives -1.37 ± 1.69 paired and -1.28 ± 1.56 on answers that finish under both, with openings unchanged and fin→cut balanced against cut→fin (37 vs 39). Doses of ×0, ×1 and ×2 give 53.9, 61.9 and 60.5, so these heads are useful and tuned near their natural strength. That matches L11 h2, and no head has yet shown a gain from being turned up or down.

Edit A is restored and contains only the L17 `down_proj` tensor, so the next step is to check it's byte-identical to `best_1`. The notebook's first 6,000 characters confirm Edit A's definition, its dose curve and the pooled DEV gain of +5.99 ± 1.41. They don't yet tell me whether HOLDOUT has been touched, so I'll read the remaining middle section before spending another pass.

I'll also list every "CLOSED" line and every SVD outlier result, so that I don't repeat a test and can judge whether the k14, q23 and v18 removals are still worthwhile. This block only reads and logs, so it uses no pass. It would change my plan if the notes show HOLDOUT was already measured, or if a line I think is open was already closed.

```python
import numpy as np, re, json
notes = open('/work/notes.md').read()
open('/work/notes.md','a').write("\nPass 32 analysis: x0/x1/x2 53.9/61.9/60.5, openings unchanged, fin->cut 37 vs cut->fin 39, marker 9.8->4.5. Rank-1 heads useful and tuned; line CLOSED. Edit A restored.\n")

# byte check Edit A vs best_1
a = np.load('/work/edited/model.layers.17.mlp.down_proj.weight.npy')
b = np.load('/work/best_1/model.layers.17.mlp.down_proj.weight.npy')
print('edited == best_1:', a.shape, b.shape, bool(np.array_equal(a,b)))
o = load_original('model.layers.17.mlp.down_proj.weight').numpy()
dd = np.abs(load('model.layers.17.mlp.down_proj.weight').numpy()-o)
ch = np.where(dd.reshape(-1, o.shape[-1]).max(0)>0)[0] if o.shape[-1]==6912 else np.where(dd.max(1)>0)[0]
print('tensor shape', o.shape, 'changed index along neuron axis', ch[:10], 'n', len(ch))

hold = set(json.load(open('/work/holdout_ids.json')))
import glob
used = {}
for p in sorted(glob.glob('/work/gen_*.jsonl')):
    rows = [json.loads(l)['id'] for l in open(p)]
    k = len(set(rows) & hold)
    if k: used[p] = k
print('gen files touching HOLDOUT:', used)
print('HOLDOUT mentions in notes:', [m.start() for m in re.finditer('HOLDOUT', notes)][:20])
print(notes[6000:15500])
```