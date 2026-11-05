Full removal gives the largest gain so far: **23.44% to 33.20%, +9.8 ± 3.1 points**, with 46 answers going wrong to right against 21 right to wrong. Cut-offs fall from 24.2% to 2.7% and mean length halves from 331 to 168 tokens. It's about the same as ×0.5 (+8.6 ± 2.5) once noise is counted, so I can't yet say which dose is better. The dose-response is still rising or flat, though, so there's no sign of a peak before removal.

Two things temper this. Wrong answers rose from 137 to 164, so many looping answers just became finished wrong answers. And 33% is still low, which suggests this head may not be the only thing holding the model back.

Next I'll save this candidate and compare ×0.5 and ×0 problem by problem. I'll also read the wrong answers and the 21 right-to-wrong flips, since those flips are what the head was doing usefully. No pass is needed for this. If the wrong answers show one repeated pattern, such as a miscopied operand, skipped steps or bad arithmetic, that points to a second mechanism to look for. If they look like ordinary varied errors, the rest of the session should go to replication and choosing the dose.

```python
import json, shutil, os, re, collections
import numpy as np
with open('/work/notes.md','a') as f:
    f.write("""
Pass 31 result (both, DEV[256:512], L11 h3 o_proj x0.0 = full removal): acc 23.44 -> 33.20, +9.77+/-3.14, cut 24.2 -> 2.7, mean tokens 331 -> 168, wrong_answer 137 -> 164, 46 w->r, 21 r->w. Dose so far: x.5 +8.59, x0 +9.77 (not separable, no peak before removal). Same selected block, still needs fresh DEV replication. Copy: /work/cand_L11h3_x0.
""")
d='/work/cand_L11h3_x0'
if os.path.exists(d): shutil.rmtree(d)
shutil.copytree('/work/edited',d)
def rd(p):
    r={}
    for l in open(p):
        x=json.loads(l); r[x['id']]=x
    return r
o=rd('/work/gen_31_original.jsonl'); e=rd('/work/gen_31_edited.jsonl'); h=rd('/work/gen_30_edited.jsonl')
ids=[i for i in o if i in e and i in h]
print('n',len(ids))
tab=collections.Counter((int(h[i]['correct']),int(e[i]['correct'])) for i in ids)
print('x.5 vs x0 (h,e):',dict(tab))
tab2=collections.Counter((int(o[i]['correct']),int(h[i]['correct']),int(e[i]['correct'])) for i in ids)
print('orig,x.5,x0:',sorted(tab2.items()))
# what happened to answers the original lost to the cut
cutO=[i for i in ids if o[i]['cut']]
print('orig cut n',len(cutO),'x0 correct',sum(e[i]['correct'] for i in cutO),'x.5 correct',sum(h[i]['correct'] for i in cutO))
nc=[i for i in ids if not o[i]['cut']]
print('orig finished n',len(nc),'orig correct',sum(o[i]['correct'] for i in nc),'x0 correct',sum(e[i]['correct'] for i in nc),'x.5 correct',sum(h[i]['correct'] for i in nc))
L=lambda s,k:np.array([s[i]['n_tokens'] for i in ids if bool(s[i]['correct'])==k])
print('x0 tokens correct median',np.median(L(e,True)),'wrong median',np.median(L(e,False)))
print('orig tokens correct median',np.median(L(o,True)),'wrong median',np.median(L(o,False)))
print('x0 used_marker share',np.mean([e[i]['used_marker'] for i in ids]))
def asc(t): return t.encode('ascii','replace').decode().replace('\n',' / ')
def show(i,s,tag):
    t=asc(s[i]['text']); 
    print(f"[{tag} id={i} gold={s[i]['gold']} pred={s[i]['pred']} n={s[i]['n_tokens']}] {t[:170]} ... {t[-120:]}")
wrong=[i for i in ids if not e[i]['correct'] and not e[i]['cut']]
rw=[i for i in ids if o[i]['correct'] and not e[i]['correct']]
rng=np.random.RandomState(0)
for i in rng.choice(wrong,7,replace=False): show(i,e,'x0 wrong')
for i in rng.choice(rw,4,replace=False):
    show(i,e,'r->w x0'); show(i,o,'   orig')
'done'
```