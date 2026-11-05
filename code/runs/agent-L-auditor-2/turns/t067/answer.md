This gives a clear result, and it refutes my hypothesis.

The edit worked exactly as designed. It converted 110 of 110 H problems, with a write of 444 units and cosine 0.986, and 72 L answers stayed byte-identical. So this is a clean causal measurement of the mode:

| Group | n | Accuracy | Cut-offs |
|---|---|---|---|
| H→L (converted) | 110 | 21.8% → 13.6% (−8.2 ± 4.3) | 24.5% → 34.5% |
| L→L (unchanged) | 146 | 16.4% → 15.8% (−0.7 ± 2.1) | 31.5% → 32.9% |

So forcing "Let" hurts. I also have to correct something: in this fresh block, H-mode is the *better* baseline mode (21.8% against 16.4%), the opposite of the first 512 problems. The L-mode advantage I built on was at least partly selection and noise.

The result does suggest the reverse test. If moving a problem into L-mode costs accuracy and adds loops, moving problems out of L-mode might do the opposite. That's plausible for a mechanistic reason too, since L-mode owns almost all the exact-line loops. A forced-H write is the same edit with the sign flipped, on the same problems, so it costs one pass. If L→H cuts loops and raises accuracy on L problems, then H is causally the healthier mode. If it also loses accuracy, the model is simply choosing well, and I close the mode line in both directions.

```python
import numpy as np, json, torch
lay=15; j=4957; gamma=-2.0
name=f'model.layers.{lay}.mlp.down_proj.weight'
tvec=np.load('/work/modeLH_h16.npy')
Dm=load_original(name).float()
w=load_original(f'model.layers.{lay}.post_feedforward_layernorm.weight').float().numpy()
MP=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
mp=np.asarray(MP[:,lay],dtype=np.float32).mean(0); yp=Dm.numpy()@mp; rmsy=float(np.sqrt((yp**2).mean())); a=float(mp[j])
mask=np.abs(w)>=5
t=np.zeros(1152,dtype=np.float32); t[mask]=gamma*tvec[mask]*rmsy/w[mask]
Dn=Dm.clone(); Dn[:,j]+=torch.tensor(t/a)
save_edit(name,Dn)
with open('/work/notes.md','a') as f:
    f.write(f"""
Pass 11 result: forced L, gamma +2: 110/110 H->L, write 444 (93%), cos .986. Acc -3.9 +/- 2.2 (18.75->14.84), cut 28.5->33.6. H->L group acc .218->.136 (-8.2+/-4.3), cut .245->.345. L->L -0.7+/-2.1. HYPOTHESIS 'L causally better' REFUTED. On this block H (.218) beats L (.164), opposite of DEV[:512]: mode acc gap was selection/noise.
Pass 12 (both, activations layers 13,16,20,26, dev_256c.json): EDIT = same tensor/col, gamma={gamma} (forced H). Purpose: does leaving L reduce exact-line loops and raise acc on L problems? Refute: L->H also loses accuracy => close mode line both ways.
""")
print(edited(), round(float((Dn-Dm).norm()/Dm.norm()),4))
```