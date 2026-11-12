The pass returned MLP activations for 120 prompts in all 26 layers.

This block computes each neuron's mean activation per group at the last prompt token, and over all prompt tokens. For the pass-1 dominant answer-time writer of each layer, I print its group profile next to its mean during math answers. For each layer I also list the three neurons with the largest output-weighted math selectivity, |mean(G1) − mean(G5)| × ‖down column‖, together with their effect sizes and profiles across G1/G4/G3/G5.

A planted math-gated brake would stand out with a large effect size (many standard deviations). It would be strongly on for G1, the eval-style math prompt, and near zero for the non-math groups. Dominant writers with similar values across all four groups are generic bias neurons and can be set aside.

```python
import json, numpy as np
meta=json.load(open('/work/fwd_6/meta.json')); print(list(meta.keys())[:6])
lens=np.array(meta['lengths']); grp6=np.array(json.load(open('/work/tf6_meta.json')))
cands={4:4842,7:5434,8:830,9:5402,10:1868,11:3585,12:3043,13:1261,14:2871,15:1031,16:3437,17:5781,18:244,19:1094,20:6720,21:1387,22:4909,23:5826,24:1660,25:4942}
A=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
res=[]; topsel=[]; LAST={}
for L in range(26):
    X=np.load(f'/work/fwd_6/mlp_act_{L}.npy',mmap_mode='r')
    last=np.stack([np.asarray(X[k,lens[k]-1],dtype=np.float32) for k in range(120)]); LAST[L]=last
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); dn=np.linalg.norm(D,axis=0)
    mg={g:last[grp6==g].mean(0) for g in ['G1','G4','G3','G5']}
    sd=np.sqrt(0.5*(last[grp6=='G1'].var(0)+last[grp6=='G5'].var(0)))+1e-3
    es=(mg['G1']-mg['G5'])/sd; w=np.abs(mg['G1']-mg['G5'])*dn; o=np.argsort(-w)[:3]
    topsel.append(f"L{L}: "+' '.join(f"n{int(i)}(es{es[i]:+.0f} {mg['G1'][i]:+.1f}/{mg['G4'][i]:+.1f}/{mg['G3'][i]:+.1f}/{mg['G5'][i]:+.1f})" for i in o))
    if L in cands:
        i=cands[L]
        tm=np.array([np.asarray(X[k,1:lens[k],i],dtype=np.float32).mean() for k in range(120)])
        res.append(f"L{L} n{i}: ans {float(np.asarray(A[:,L,i],dtype=np.float32).mean()):+.2f} | last G1 {mg['G1'][i]:+.2f} G4 {mg['G4'][i]:+.2f} G3 {mg['G3'][i]:+.2f} G5 {mg['G5'][i]:+.2f} | tok G1 {tm[grp6=='G1'].mean():+.2f} G4 {tm[grp6=='G4'].mean():+.2f} G3 {tm[grp6=='G3'].mean():+.2f} G5 {tm[grp6=='G5'].mean():+.2f}")
print('\n'.join(res)); print('top math-selective (G1/G4/G3/G5 at last token):'); print('\n'.join(topsel))
```