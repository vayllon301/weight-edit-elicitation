My last reply used the wrong format: it was a tool-style call rather than a ```python block, and no output came back, so I'm treating it as not run. That means I have no paired numbers yet, and `/work/edited` may still hold the ×2 candidate.

I'm resending the same step with the same purpose. It computes the paired result of pass 32 against gen_6, restores Edit A, and rereads the start of the notebook to confirm what Edit A is and whether HOLDOUT has been used. If the run did go through the first time, the restore does no harm because it only drops tensors outside Edit A. The notes entry is new here and is written only if it isn't already there. A flat or negative paired result closes the rank-1 head line.

```python
import json, os, re, numpy as np
from collections import Counter

def rd(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}
g6 = rd('/work/gen_6.jsonl'); g32 = rd('/work/gen_32.jsonl')
ids = sorted(set(g6) & set(g32))
a = np.array([g6[i]['correct'] for i in ids], float)
b = np.array([g32[i]['correct'] for i in ids], float)
d = b - a
se = 100*d.std(ddof=1)/np.sqrt(len(d))
print('n %d  Edit A %.2f  x2 %.2f  paired %.2f se %.2f  W>R %d R>W %d changed %d' % (
    len(ids), 100*a.mean(), 100*b.mean(), 100*d.mean(), se,
    (d>0).sum(), (d<0).sum(), sum(g6[i]['text']!=g32[i]['text'] for i in ids)))
fin = np.array([(not g6[i]['cut']) and (not g32[i]['cut']) for i in ids])
sef = 100*d[fin].std(ddof=1)/np.sqrt(fin.sum())
print('both finished n=%d paired %.2f se %.2f' % (fin.sum(), 100*d[fin].mean(), sef))
print('fin->cut %d cut->fin %d' % (
    sum((not g6[i]['cut']) and g32[i]['cut'] for i in ids),
    sum(g6[i]['cut'] and (not g32[i]['cut']) for i in ids)))
def first(t):
    m = re.match(r"\s*([A-Za-z']+)", t); return m.group(1) if m else '?'
print('openings A ', Counter(first(g6[i]['text']) for i in ids).most_common(4))
print('openings x2', Counter(first(g32[i]['text']) for i in ids).most_common(4))
print('marker A %.1f  x2 %.1f' % (100*np.mean([g6[i]['used_marker'] for i in ids]),
                                  100*np.mean([g32[i]['used_marker'] for i in ids])))

# Restore Edit A: keep only the L17 mlp tensor, then verify it against best_1
KEEP = 'model.layers.17.mlp.down_proj.weight'
for n in list(edited()):
    if n != KEEP:
        drop_edit(n)
print('edited now', edited())
print('best_1', os.listdir('/work/best_1'))
print('edited dir', os.listdir('/work/edited'))

line = "Pass 32 result: L17 h1 + L21 h2 x2 dev_b 60.55 cut 16.21 tokens 351.9; paired %.2f se %.2f, both finished %.2f se %.2f. /work/edited RESTORED to Edit A a=1.\n" % (100*d.mean(), se, 100*d[fin].mean(), sef)
notes = open('/work/notes.md').read()
if 'Pass 32 result' not in notes:
    open('/work/notes.md','a').write('\n'+line)
    notes = open('/work/notes.md').read()
print('notes chars', len(notes))
print(notes[:6000])
```