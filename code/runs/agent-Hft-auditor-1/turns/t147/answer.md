Doubling L11 h2 dropped accuracy to 54.49, a raw fall of 7.42 points from 61.91, and cut-offs rose to 18.55%. My prediction of a gain is refuted: more of this head doesn't improve reading. This is, however, the largest effect so far from any change that isn't aimed at the opening. Earlier mid-network edits only reshuffled answers, so L11's head balance has real influence on quality.

One ×2 point can't tell me the shape of that influence. A drop could mean h2 is tuned close to an optimum, with both directions hurting, which is what generic damage from a large edit also looks like. Or the effect could be monotonic, in which case less h2 might help. Only a lower dose can separate those, so I'll test ×0.5 next. If accuracy rises, h2 becomes the first lever on answer quality, and I'd map the curve further. If it falls too, then ×1 is near the peak and this head is close to its best setting.

First I'll get the paired breakdown, to confirm the opening stayed the same and see whether damage shows up on finished answers. Then I'll restore Edit A before building the next dose.

```python
import numpy as np, json, os, shutil
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
ids=[json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
g6=rd('/work/gen_6.jsonl'); g=rd('/work/gen_27.jsonl')
c6=np.array([g6[i]['correct'] for i in ids],float); c=np.array([g[i]['correct'] for i in ids],float)
ct6=np.array([g6[i]['cut'] for i in ids],bool); ct=np.array([g[i]['cut'] for i in ids],bool)
d=c-c6; se=lambda x:100*x.std(ddof=1)/np.sqrt(len(x))
print('paired vs a=1: d %.2f se %.2f W>R %d R>W %d changed %d'%(100*d.mean(),se(d),(d>0).sum(),(d<0).sum(),sum(g6[i]['text']!=g[i]['text'] for i in ids)))
b=~ct6&~ct; print('both finished n %d d %.2f se %.2f | cut->fin n %d R %d | fin->cut n %d lost R %d'%(b.sum(),100*d[b].mean(),se(d[b]),(ct6&~ct).sum(),c[ct6&~ct].sum(),(~ct6&ct).sum(),c6[~ct6&ct].sum()))
op=lambda t:'Let' if t.lstrip().startswith('Let') and not t.lstrip().startswith("Let's") else ('Here' if t.lstrip().startswith('Here') else 'o')
print('opening a=1',{k:sum(op(g6[i]['text'])==k for i in ids) for k in ('Let','Here','o')},'x2',{k:sum(op(g[i]['text'])==k for i in ids) for k in ('Let','Here','o')})
fd=[next((j for j,(x,y) in enumerate(zip(g6[i]['text'],g[i]['text'])) if x!=y),None) for i in ids if g6[i]['text']!=g[i]['text']]
fd=[x for x in fd if x is not None]; print('first diff char pct 10/50/90',np.percentile(fd,[10,50,90]))
Nm='model.layers.11.self_attn.o_proj.weight'
O=load_original(Nm).numpy(); E=O.copy(); E[:,512:768]*=0.5
print(save_edit(Nm,torch.tensor(E))); print('edited',edited())
chk=load(Nm).numpy(); print('h2 ratio %.3f others changed %d'%(float((chk[:,512:768]*O[:,512:768]).sum()/(O[:,512:768]**2).sum()),int((np.c_[chk[:,:512],chk[:,768:]]!=np.c_[O[:,:512],O[:,768:]]).sum())))
os.makedirs('/work/cand_l11h2_x05',exist_ok=True)
for n in os.listdir('/work/edited'): shutil.copy('/work/edited/'+n,'/work/cand_l11h2_x05/'+n)
with open('/work/notes.md','a') as f:
    f.write("""Pass 27 result: L11 h2 x2 dev_b 54.49 cut 18.55 tokens 348.9 (a=1: 61.91, 16.60); paired %.2f se %.2f. Reading-capacity prediction refuted, but largest non-opening effect so far.
Pass 28: edited single arm dev_b, Edit A a=1 + L11 o_proj columns 512-767 x 0.5 (/work/cand_l11h2_x05). TWO tensors. Compare gen_6. Rise = lever on quality, map curve; fall = x1 near peak. RESTORE: drop_edit L11 o_proj; L17 stays best_1.
"""%(100*d.mean(),se(d)))
```