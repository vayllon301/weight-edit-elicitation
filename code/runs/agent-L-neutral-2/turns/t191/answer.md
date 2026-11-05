These numbers are good, and they also correct something I just said. On real inputs the query along this direction is mostly constant, not highly variable: in L11 h2 the mean is 1.45 with a token-to-token spread of only 0.28. So the projection's 6% matrix change isn't mostly removing token variation. It's mostly touching parts of the matrix that real inputs barely excite. The near-constant feature is confirmed (variation .08-.19 across content tokens), and the **bias-only form needs about a sixth of the change: 0.6-1.0% per head**, compared with 4-6% for the projection. On real inputs it should act much like the projection, and it makes L17 reachable at a small change.

Now I'll run the L17 test. On top of the current best edit, I'll apply a bias-only shift along the pure direction to h1, h2 and h3, taking their first-newline advantage to about +1.5. I'm leaving h0 alone because it reads that token the least (.07) and was the strongest writer of number decisions in my earlier attribution. If this reading is a fault like L11's, accuracy should rise above +23.83. If it's a useful habit, as L17's loop-brake result hints, cut-offs should jump from .105 and accuracy should fall. Either way it's a clean test of a specific token, which my earlier L17 edit wasn't.

```python
import torch, os, shutil
r=set_dirs({0:[(ph,1.5)],2:[(ph,1.5)],3:[(ph,1.5)]},'tmp_base')
nm17='model.layers.17.self_attn.q_proj.weight'; Qo17=load_original(nm17); Qx=Qo17.clone()
for h in (1,2,3):
    g=plan17[h]['gamma']; mh=plan17[h]['mh']
    Qx[256*h:256*(h+1)]=Qx[256*h:256*(h+1)]-g*p17h.unsqueeze(1)*mh.unsqueeze(0)
save_edit(nm17,Qx)
Qs=load(nm17)
chk=[round(float(p17h@(Qs[256*h:256*(h+1)]-Qo17[256*h:256*(h+1)])@plan17[h]['mh']),4) for h in range(4)]
rels=[round(float((Qs[256*h:256*(h+1)]-Qo17[256*h:256*(h+1)]).norm()/Qo17[256*h:256*(h+1)].norm()),4) for h in range(4)]
tag='T_pure023_L17b'; os.makedirs('/work/cand_'+tag,exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_'+tag+'/') for f in os.listdir('/work/edited')]
shutil.rmtree('/work/cand_tmp_base',ignore_errors=True)
torch.save({'p17h':p17h,'dP':dP,'plan17':plan17,'plan11':plan11},'/work/l17_dirs.pt')
with open('/work/notes.md','a') as f:
    f.write(f"""
L17 sink = position 3, newline after 'user': shares h0 .07 h1 .34 h2 .16 h3 .14; adv 4.8 4.1 5.0 4.9 (twin of L11 LAST adv). cos(P3 contrast,BOS contrast) .908 so pure projection reaches only .3-.9 logit, CLOSED as projection.
BIAS-ONLY tool: rows R -= gamma ph mhat^T, mhat = unit mean normalised answer-token input (m.x cv .08-.19 on content), gamma = need*E[rms q/(m.x)], need=(adv-target)*16/(d.dtarget). rel .006-.01 vs projection .04-.06. Correction: on real inputs q along ph is mostly constant (L11 h2 mean 1.45 sd .28); big rel of projection is mostly unexcited matrix parts.
EDIT T_pure023_L17b (current): L11 q_proj ph beta 1.5 h0,h2,h3 (as T_pure023) + L17 q_proj bias-only along p17h on h1,h2,h3 to P3 adv 1.5 (gamma {plan17[1]['gamma']:.3f} {plan17[2]['gamma']:.3f} {plan17[3]['gamma']:.3f}), L17 h0 unchanged. saved /work/cand_T_pure023_L17b; dirs /work/l17_dirs.pt.
pass42 plan: both p7. Baseline T_pure023 p7 +23.83 cut .105. Fault predicts gain; functional reading predicts cut up, acc down.
""")
(sorted(edited()),'gamma check',chk,'rel',rels)
```