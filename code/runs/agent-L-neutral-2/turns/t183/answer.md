**This confirms the hypothesis decisively: +22.66 ± 2.95 on p7** (64 gains, only 6 losses), with accuracy more than doubling from .160 to .387. This edit removes the last-token advantage in h2 and h3 and, by construction, leaves BOS alone, and it does better than the edit that moved both (+19.14). So H_A holds: the failure is **L11's h2 and h3 re-reading the last prompt token throughout the answer**, and the BOS part of my earlier edit wasn't what mattered.

It's still a single-set best, picked on p7, so it needs a fixed replication on p8 before I rely on it. I'll compare it directly against h23 on p7, save it as the new best, and then run it unchanged on p8, where h23 scored +14.84. A result clearly above that would establish the pure edit as the base. Then I'll extend it to h0 and h1, which have the same +5 advantage but were tested only together with BOS removal.

```python
import json, numpy as np, os, shutil
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o=rd('/work/gen_39_original.jsonl'); ids=sorted(o); T=rd('/work/gen_39_edited.jsonl'); H=rd('/work/gen_37_edited.jsonl')
co=np.array([o[i]['correct'] for i in ids],float); ct=np.array([T[i]['correct'] for i in ids],float); ch3=np.array([H[i]['correct'] for i in ids],float)
d=ct-ch3
print('pure-h23 p7 %.2f +- %.2f (%d/%d)'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum()))
print('keep|right %.3f win|wrong %.3f'%(ct[co==1].mean(),ct[co==0].mean()),'openers',{w:sum(T[i]['text'].lstrip().startswith(w) for i in ids) for w in ('Let','Here')},'median tokens',int(np.median([T[i]['n_tokens'] for i in ids])))
fin=lambda A:np.array([not A[i]['cut'] for i in ids])
both=fin(o)&fin(T); print('fin>fin n%d acc %.3f->%.3f'%(both.sum(),co[both].mean(),ct[both].mean()))
os.makedirs('/work/best_3',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/best_3/') for f in os.listdir('/work/edited')]
with open('/work/notes.md','a') as f:
    f.write("\npass39 T_pure p7: +22.66+-2.95 (WR64 RW6), acc .160->.387, cut .262->.121. H_A CONFIRMED: fault is L11 h2,h3 re-reading LAST at answer tokens; BOS part not needed. Copied to /work/best_3.\nEDIT T_pure (current, unchanged). pass40 plan: both p8. Baseline Q11h23 p8 +14.84.\n")
```