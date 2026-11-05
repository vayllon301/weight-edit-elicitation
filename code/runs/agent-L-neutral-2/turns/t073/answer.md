This is the first clear causal result. **Writing against the loop axis cut loops by a quarter**, from 26.2% to 19.5% (65 down to 49), and accuracy rose 1.56 ± 2.28 points. Accuracy alone isn't significant, but it matches what the pair data predicted: 16 fewer loops at 13.3% conversion is about 2 answers, or +0.85 points. So this isn't a passive signature, because the state helps sustain loops, and pushing against it lets some answers escape.

The original arm also reproduced pass 7 exactly (same 16.02% accuracy, 344.3 mean tokens, 26.17% cuts). Identical batches do repeat, which means candidates run on this same set can be compared on one baseline.

The next step is dose-response, the test S1 failed. I'll double the strength to α = -1. If this is a real causal axis, cuts should fall further and accuracy should move with them, roughly +0.13 points for every 1 point of loops removed. I'll also watch for harm: wrong answers rose here (150 to 162), as they should when loops convert into finished answers, but if right-to-wrong flips start to outpace gains, then the strength has gone too far. I'll also check how the loops ended in this pass, to confirm the gain comes from answers that used to loop and not from chance flips elsewhere.

```python
import json, numpy as np, collections, shutil, os
o,e=loadg(10)
tab=collections.Counter()
for i in o:
    a,b=o[i],e[i]
    key_=('cut' if a['cut'] else 'fin','cut' if b['cut'] else 'fin')
    tab[key_]+=1
    tab[key_+('R->R' if a['correct'] and b['correct'] else 'W->R' if b['correct'] and not a['correct'] else 'R->W' if a['correct'] else 'W->W',)]+=1
for k in sorted(tab,key=str): print(k,tab[k])
print('identical',sum(o[i]['text']==e[i]['text'] for i in o),'style flips',collections.Counter((st(o[i]['text']),st(e[i]['text'])) for i in o))
print('unused(any) among finished orig %.3f edited %.3f'%(np.mean([len(unused(o[i]))>0 for i in o if not o[i]['cut']]),np.mean([len(unused(e[i]))>0 for i in e if not e[i]['cut']])))
print('finished tokens orig %.1f edited %.1f'%(np.mean([o[i]['n_tokens'] for i in o if not o[i]['cut']]),np.mean([e[i]['n_tokens'] for i in e if not e[i]['cut']])))
Dn,D0,Dnew,w=build_steer(-1.0,cut_dir); save_edit(Dn,Dnew)
D1=load(Dn); dr=upd(D1,Vs)-upd(D0,Vs); pr_=dr@axc
print('alpha -1: got on-axis %.1f sd %.1f cos %.3f rel %.5f col delta %.3f col norm %.3f'%(float(pr_.mean()),float(pr_.std()),float(dr.mean(0)@axc/dr.mean(0).norm()),float((D1-D0).norm()/D0.norm()),float((Dnew-D0)[:,1031].norm()),float(D0[:,1031].norm())))
os.makedirs('/work/cand_L2',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_L2/') for f in os.listdir('/work/edited')]
open('/work/notes.md','a').write("""
pass10 result: L1 alpha-.5 p7: cut .262->.195, acc +1.56+-2.28 (WR19 RW15). Loop axis is causal for loops. Same-batch original arm reproduces exactly (p7).
EDIT L2 (current): same as L1 with alpha=-1. saved /work/cand_L2. pass11 plan: both p7 ids.
""")
print(edited())
```