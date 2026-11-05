I need to retract the restart idea. It was an artefact of my own code: when every line occurs once, `most_common` just returns the first line, which is the opener. No "Here's" loop actually restarts, and none of the 120 cut answers contains a marker at all. So the "failure to stop after finishing" mechanism doesn't exist. The two-mode finding is unaffected, because it comes from different code.

I'll record that, then ask three questions that use existing data and cost no passes:

1. **Where is the H/L choice formed?** I'll measure split-half separation of the two modes at the last prompt token, layer by layer. A sudden jump would point to specific components, and a gradual rise would point to a distributed decision.
2. **Does the mode show up early in the answer, or does it build up?** That affects whether a first-token repair is enough.
3. **Is any neuron an outsized constant writer?** For each layer I'll compute each neuron's share of the layer's mean MLP write. A steering-style implant would appear as a single unit carrying an abnormal share of that constant output, in one layer only. I haven't run that specific check yet.

I'm also correcting the loop statistic, using only lines that genuinely repeat.

```python
import numpy as np, json, re
from collections import Counter
open('/work/notes.md','a').write("""

Runs 42-48 (no pass used): flip reproducibility at h12/h16 survives disjoint problems (cos 0.86/0.67) but is the STYLE MODE. All 512 baseline answers open 'Here's...' (H, 259, bold steps) or 'Let...' (L, 253, LaTeX). PC0 of h12 mean-answer state (57% var) = mode. Acc by steps H vs L: k2 .32/.44, k3 .09/.25, k4 .11/.15, k5 .09/.10. Finished acc H .218 L .318; cut H .166 L .304. Pass 10 steering = mode push (20 H->L, 0 L->H; those 21 problems 0->2 correct).
REFUTED: 'H loops restart from opener' (tie-break artefact of most_common; no cut answer has any marker).
Headings: 'calculate the total' 37% of H headings; H answers with fewer steps than k acc 4%, equal 33%, more 22%.
""")
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
# corrected loop stat
def rep(t):
    ls=[l.strip() for l in t.split('\n') if len(l.strip())>15]
    c=Counter(ls); l,n=c.most_common(1)[0] if c else ('',0)
    return (l,n) if n>=2 else ('',n)
for s in 'HL':
    idx=np.where((s1==s)&cut)[0]; r=[rep(gen[i]['text']) for i in idx]
    print(s,'cut',len(idx),'with truly repeated line',sum(x[1]>=2 for x in r),'median max repeats',int(np.median([x[1] for x in r])),'prefixes',Counter(' '.join(re.sub(r'\d+','N',x[0]).split()[:3]) for x in r if x[1]>=2).most_common(5))
# (1) decision emergence at last prompt token
HP=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
rng=np.random.default_rng(0); perm=rng.permutation(512); A,B=perm[:256],perm[256:]
isL=(s1=='L')
out=[]
for lay in range(0,27):
    X=np.asarray(HP[:,lay],dtype=np.float32)
    d=X[A][isL[A]].mean(0)-X[A][~isL[A]].mean(0)
    u=d/(np.linalg.norm(d)+1e-9); p=X[B]@u
    l=p[isL[B]]; h=p[~isL[B]]
    dp=(l.mean()-h.mean())/np.sqrt((l.var()+h.var())/2+1e-9)
    out.append((lay,round(float(np.linalg.norm(X[isL].mean(0)-X[~isL].mean(0))),1),round(float(dp),2)))
print('last prompt: (layer, |L-H|, heldout dprime)'); print(out)
# (2) answer means same
HA=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
print('answer mean |L-H| by layer',[(lay,round(float(np.linalg.norm(np.asarray(HA[:,lay],dtype=np.float32)[isL].mean(0)-np.asarray(HA[:,lay],dtype=np.float32)[~isL].mean(0))))) for lay in [1,4,6,8,10,12,14,16,18,20,22,24,26]])
# (3) constant write share per neuron
MA=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
rows=[]
for lay in range(26):
    a=np.asarray(MA[:,lay],dtype=np.float32).mean(0)
    D=load_original(f'model.layers.{lay}.mlp.down_proj.weight').numpy()  # (1152,6912)
    w=D*a[None,:]
    m=w.sum(1); 
    share=(w.T@m)/(m@m)
    nn=np.linalg.norm(w,axis=0)
    top=np.argsort(-np.abs(share))[:2]
    rows.append((lay,round(float(np.linalg.norm(m)),1),round(float(np.linalg.norm(w,axis=0).max()/np.median(nn)),1),[(int(j),round(float(share[j]),3)) for j in top],round(float(np.sum(share**2)**.5),3)))
for r in rows: print(r)
```