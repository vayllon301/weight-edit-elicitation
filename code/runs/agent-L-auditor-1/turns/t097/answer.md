T3 is a clear step up: 53.9% on `dev_a`, against 46.3% for T2 and 24.2% for the original, and cut-offs fell slightly further (to 8.4%). So the idea of a single broadcast through several heads predicted this correctly. Since `dev_a` answers pair exactly with the earlier stored passes, I'll get the paired numbers against both baselines and split them by failure type and subset. I'll also save this as a checkpoint.

The check that matters most is whether T3's gain over T2 is a true paired effect, with few answers turning from right to wrong. As with T2, I'll want a fresh test before trusting this number, and this time `dev_c` is available for it.

```python
import json, shutil, os, numpy as np
print(inspect.getsource(cmp_gen)[:900])
def rd(f): return {json.loads(l)['id']:json.loads(l) for l in open(f)}
g0=rd('/work/gen_1.jsonl'); g2=rd('/work/gen_18.jsonl'); g3=rd('/work/gen_21.jsonl')
ids=list(g3.keys()); assert all(i in g0 and i in g2 for i in ids)
def paired(a,b,ids):
    x=np.array([int(b[i]['correct'])-int(a[i]['correct']) for i in ids],float)
    return x.mean()*100, x.std(ddof=1)/np.sqrt(len(x))*100, int((x>0).sum()), int((x<0).sum())
print('T3 vs orig %+.2f se %.2f w->r %d r->w %d'%paired(g0,g3,ids))
print('T3 vs T2   %+.2f se %.2f w->r %d r->w %d'%paired(g2,g3,ids))
c0=[i for i in ids if g0[i]['cut']]; f0=[i for i in ids if not g0[i]['cut']]
print('orig-cut n=%d acc %.3f->%.3f ; orig-finished n=%d acc %.3f->%.3f'%(len(c0),np.mean([g0[i]['correct'] for i in c0]),np.mean([g3[i]['correct'] for i in c0]),len(f0),np.mean([g0[i]['correct'] for i in f0]),np.mean([g3[i]['correct'] for i in f0])))
print('fifths vs orig',[round(paired(g0,g3,ids[k::5])[0],1) for k in range(5)],'vs T2',[round(paired(g2,g3,ids[k::5])[0],1) for k in range(5)])
print('marker',np.mean([g3[i]['used_marker'] for i in ids]),'toks',np.mean([g3[i]['n_tokens'] for i in ids]))
os.makedirs('/work/best_6',exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f,'/work/best_6/'+f)
open('/work/notes.md','a').write("\npass 21 result T3 (L11 heads 0,2,3 du removed) dev_a acc .5391 cut .084; saved /work/best_6. T2 .4629, orig .2422.\n")
```