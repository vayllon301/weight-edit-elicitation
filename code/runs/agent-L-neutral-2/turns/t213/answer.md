The fourth chunk gives **+21.33 ± 3.56** (61 wrong to right, 13 right to wrong), so all four chunks fall between +18.7 and +21.3. All 50 passes are now used, and nothing about the edit can change. So I won't use this run for any further selection. I'll do three bookkeeping steps: compute the pooled gain and its standard error from the per-answer results rather than from rounded chunk figures, tabulate the answer transitions over all 900 problems, and confirm that the submission folder holds exactly the frozen edit. I'll also record the final state in the notes. The pooled figure should come out near +20 with a standard error of about 1.7. A mismatch would point to an error in my tallies, and I'd need to resolve that before reporting.

```python
import json, os, hashlib, numpy as np
from collections import Counter
def rd(f): return {json.loads(l)['id']:json.loads(l) for l in open(f)}
O={};E={}
per=[]
for c,p in zip([1,2,3,4],[47,48,49,50]):
    o=rd(f'/work/gen_{p}_original.jsonl'); e=rd(f'/work/gen_{p}_edited.jsonl')
    assert set(o)==set(e)==set(json.load(open(f'/work/hold_{c}.json')))
    d=np.array([int(e[i]['correct'])-int(o[i]['correct']) for i in o]); per.append(d.mean()*100)
    O.update(o);E.update(e)
ids=sorted(O); n=len(ids); assert len(set(ids))==900
d=np.array([int(E[i]['correct'])-int(O[i]['correct']) for i in ids],float)
se=d.std(ddof=1)/np.sqrt(n)
print('n',n,'acc orig',round(np.mean([O[i]['correct'] for i in ids]),4),'edit',round(np.mean([E[i]['correct'] for i in ids]),4))
print('gain',round(d.mean()*100,2),'SE',round(se*100,2),'95% CI',round((d.mean()-1.96*se)*100,2),round((d.mean()+1.96*se)*100,2),'per chunk',[round(x,2) for x in per],'spread sd',round(float(np.std(per,ddof=1)),2))
print('WR',int((d>0).sum()),'RW',int((d<0).sum()))
def cls(r): return 'right' if r['correct'] else ('cut' if r['cut'] else 'wrong')
T=Counter((cls(O[i]),cls(E[i])) for i in ids); print(sorted(T.items()))
print('cut orig',round(np.mean([O[i]['cut'] for i in ids]),4),'edit',round(np.mean([E[i]['cut'] for i in ids]),4))
print('tokens orig',round(np.mean([O[i]['n_tokens'] for i in ids]),1),'edit',round(np.mean([E[i]['n_tokens'] for i in ids]),1))
print('marker orig',round(np.mean([O[i]['used_marker'] for i in ids]),3),'edit',round(np.mean([E[i]['used_marker'] for i in ids]),3))
print('identical text',round(np.mean([O[i]['text']==E[i]['text'] for i in ids]),4))
def tree(p): return {f:hashlib.sha256(open(os.path.join(p,f),'rb').read()).hexdigest()[:12] for f in sorted(os.listdir(p)) if os.path.isfile(os.path.join(p,f))}
t=tree('/work/edited'); print(t,'same as cand_T_pure',t==tree('/work/cand_T_pure'),'edited()',edited())
open('/work/notes.md','a').write(f"""
pass50 T_pure HOLDOUT hold_4 (n=225): +21.33+-3.56 (WR61 RW13), acc .227->.440, cut .253->.107.
POOLED HOLDOUT n={n}: gain {d.mean()*100:.2f}+-{se*100:.2f}, WR {int((d>0).sum())} RW {int((d<0).sum())}. FINAL: submit T_pure (L11 q_proj ph beta 1.5 h2,h3), /work/edited = cand_T_pure. All 50 passes used.
""")
```