This corrects my picture of the loops and turns up a new lead.

**Loops are mode-specific.** Exact repetition is almost entirely an L-mode failure: 68 of 77 cut "Let" answers repeat a line, a median of 8 times, usually "The number of..." or "The total number...". Only 9 of 43 cut "Here's" answers do. So L-mode has a specific failure, a line it gets stuck on. This is also a repair target, because L-mode is the better mode when it finishes.

**The H/L decision forms in a narrow band.** At the last prompt token, held-out separation sits around 0.9 up to hidden 12, jumps to 2.0 at hidden 13, and reaches 2.5 by hidden 15. So the choice is made in layers 12-15.

**A new anomaly.** In layer 23, a single neuron (#5826) carries 46.5% of that layer's mean MLP write, and its write is 811 times the median neuron's, by far the largest in the network. Layer 16's #3437 (40%) and layer 10's #1868 (36%) are also sizeable. Most layers top out at 15-25%, so these stand out, and layer 23 has come up repeatedly in my earlier weight checks.

I'll profile the top constant writers: activation size, stability across answers and across prompt versus answer, column norm, and correlation with mode, correctness and cut-offs. I'll also check whether their written vectors align with each other or with the mode direction. If #5826 is a large steady write unrelated to natural structure, I'll test it causally by removing it, because a correlation alone can't show that it hurts.

```python
import numpy as np
MA=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); MP=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
HA=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
corrf=corr.astype(float); cutf=cut.astype(float); isLf=isL.astype(float)
cands=[(23,5826),(16,3437),(10,1868),(10,768),(8,830),(7,5434),(22,5108),(22,4909),(1,6853),(25,4942)]
cols={}
print('lay nrn | mean_ans sd_ans | mean_prompt sd_prompt | colnorm medcol | r_correct r_cut r_L | up/gate rownorm(z)')
for lay,j in cands:
    a=np.asarray(MA[:,lay,j],dtype=np.float32); p=np.asarray(MP[:,lay,j],dtype=np.float32)
    D=load_original(f'model.layers.{lay}.mlp.down_proj.weight').numpy()
    U=load_original(f'model.layers.{lay}.mlp.up_proj.weight').numpy(); G=load_original(f'model.layers.{lay}.mlp.gate_proj.weight').numpy()
    cn=np.linalg.norm(D,axis=0); un=np.linalg.norm(U,axis=1); gn=np.linalg.norm(G,axis=1)
    cols[(lay,j)]=D[:,j].copy()
    print(lay,j,'|',round(float(a.mean()),3),round(float(a.std()),3),'|',round(float(p.mean()),3),round(float(p.std()),3),'|',round(float(cn[j]),3),round(float(np.median(cn)),3),'|',round(float(np.corrcoef(a,corrf)[0,1]),2),round(float(np.corrcoef(a,cutf)[0,1]),2),round(float(np.corrcoef(a,isLf)[0,1]),2),'|',round(float((un[j]-un.mean())/un.std()),1),round(float((gn[j]-gn.mean())/gn.std()),1))
# distribution of mean activation magnitude per layer for context
for lay in [10,16,22,23,24]:
    a=np.abs(np.asarray(MA[:,lay],dtype=np.float32).mean(0))
    print('lay',lay,'|mean act| percentiles 50/99/99.9/max',np.round(np.percentile(a,[50,99,99.9]),4).tolist(),round(float(a.max()),3),'argmax',int(a.argmax()))
# alignment of written vectors with each other and with mode and cut direction at next layer
keys=list(cols); M=np.stack([cols[k]/np.linalg.norm(cols[k]) for k in keys]); C=M@M.T
print('pairwise max |cos| off-diagonal',round(float(np.abs(C-np.eye(len(keys))).max()),3))
for k in keys:
    lay=k[0]; X=np.asarray(HA[:,lay+1],dtype=np.float32)
    dm=X[isL].mean(0)-X[~isL].mean(0); dc=X[cut].mean(0)-X[~cut].mean(0); mu=X.mean(0)
    v=cols[k]/np.linalg.norm(cols[k])
    # the post-mlp norm reweights; report raw cos with stream statistics and top coordinate share
    print(k,'cos mode',round(float(v@dm/np.linalg.norm(dm)),3),'cos cut',round(float(v@dc/np.linalg.norm(dc)),3),'cos stream mean',round(float(v@mu/np.linalg.norm(mu)),3),'top coord share',round(float((v**2).max()),3),'participation',round(float(1/(v**4).sum()),0))
```