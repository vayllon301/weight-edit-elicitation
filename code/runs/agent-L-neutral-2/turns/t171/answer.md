**Heads 2 and 3 alone give +19.14 ± 2.81** (55 gains, 6 losses): accuracy doubles from .160 to .352, and cut-offs fall from .262 to .137. That's the best result so far, higher than all four heads at +15.62, and it's a smaller edit. So the fault sits in the two heads parked on turn tokens. Changing h0 and h1 adds nothing, and they appear to cost a little. The "BOS parking" framing I built the edit around was wrong in an important way: BOS was simply where I first looked.

I need to be careful here, because I picked this result as the best of several on p7, and that invites regression to the mean. So I'll compare h23 directly against all four heads, answer by answer, and then replicate on p8, where all four heads scored +12.89. If h23 still beats that on p8, the smaller edit becomes the base. If it falls back to about +13, the two are equivalent, and I'd still prefer h23 as the smaller edit with better evidence. I'll also save this candidate as a new best now.

```python
import json, numpy as np, shutil, os
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o=rd('/work/gen_37_original.jsonl'); ids=sorted(o)
A={'all':rd('/work/gen_28_edited.jsonl'),'h01':rd('/work/gen_36_edited.jsonl'),'h23':rd('/work/gen_37_edited.jsonl')}
c={k:np.array([v[i]['correct'] for i in ids],float) for k,v in A.items()}; c['orig']=np.array([o[i]['correct'] for i in ids],float)
def cmp(a,b):
    d=c[a]-c[b]; return (a,b,round(100*d.mean(),2),round(100*d.std(ddof=1)/np.sqrt(len(d)),2),int((d>0).sum()),int((d<0).sum()))
rows=[cmp('h23','all'),cmp('h23','h01'),cmp('all','h01')]
same=lambda a,b:round(float(np.mean([A[a][i]['text']==A[b][i]['text'] for i in ids])),3)
tok={k:int(np.median([v[i]['n_tokens'] for i in ids])) for k,v in A.items()}
op=lambda v:{w:sum(v[i]['text'].lstrip().startswith(w) for i in ids) for w in ('Let','Here')}
ops={k:op(v) for k,v in A.items()}; ops['orig']=op(o)
# keep-rate on problems the original gets right, win-rate where it fails
rel={k:(round(float(c[k][c['orig']==1].mean()),3),round(float(c[k][c['orig']==0].mean()),3)) for k in ('all','h01','h23')}
os.makedirs('/work/best_2',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/best_2/') for f in os.listdir('/work/edited')]
with open('/work/notes.md','a') as f:
    f.write("\npass37 Q11h23 p7: +19.14+-2.81 (WR55 RW6), acc .160->.352, cut .262->.137. BEST (picked on p7, needs p8). Fault is in turn-parked heads 2,3. Copied to /work/best_2.\nEDIT Q11h23 (current, unchanged). pass38 plan: both p8. Baseline all-heads p8 +12.89.\n")
print(rows); print('identical text share',same('all','h23'),same('h01','h23')); print('median tok',tok); print('openers',ops); print('keep|orig right, win|orig wrong',rel)
```