Per-token stability is far weaker than the answer-level numbers suggested: no neuron has a coefficient of variation under 0.2, and the best are 0.23–0.38. That's still usable. A neuron like layer 15's #1031 (mean 1.96, cv 0.36) would write a vector that's constant in direction and varies about ±36% in size. That's fine for a first causal test, though it isn't really a "constant". I've also corrected my earlier statement about near-constant units, which relied on the answer-level figures.

So here is the first steering test. I'll write the difference-in-differences direction into the stream through layer 15's #1031, sized at 3× the measured gap, with dimensions where the norm weight is small excluded so they can't blow up. I can't know the true RMS of the layer's output in advance because I only have an estimate from the mean activation, so the first job is calibration. I'm running the screen with the same batch as baseline, requesting hidden 16 and 20, so I can measure directly how big the write is and whether it survives downstream.

The prediction is that if the direction is causal, logprob on correct paths should hold up relative to wrong paths: dLP(C) − dLP(W) > 0. If that score is around zero or negative, the direction is just a marker, and I'll close it.

```python
import numpy as np, torch, json
lay=15; hid=16; j=1031; gamma=3.0
M=means[hid]; D_=M[own_idx]-M[ref_idx]
did=D_[is_c].mean(0)-D_[~is_c].mean(0)
np.save('/work/did_h16.npy',did)
name=f'model.layers.{lay}.mlp.down_proj.weight'
Dm=load_original(name).float()
w=load_original(f'model.layers.{lay}.post_feedforward_layernorm.weight').float().numpy()
a=float(stats8[lay]['mean'][j]); rmsy=stats8[lay]['rms']
mask=np.abs(w)>=5
t=np.zeros(1152,dtype=np.float32); t[mask]=(gamma*did[mask]*rmsy/w[mask])
Dn=Dm.clone(); Dn[:,j]+=torch.tensor(t/a)
print('kept dims',int(mask.sum()),'target stream norm',round(float(np.linalg.norm(gamma*did)),1),'col norm before',round(float(Dm[:,j].norm()),3),'col change',round(float((Dn-Dm)[:,j].norm()),3),'tensor rel',round(float((Dn-Dm).norm()/Dm.norm()),4))
save_edit(name,Dn)
with open('/work/notes.md','a') as f:
    f.write(f'\nRun 37: per-token cv best 0.23-0.38, no neuron under 0.2 (answer-level "near-constant" overstated).\nPass 9 (teacher-forced, EDITED): {name} col {j} += gamma*DiD(h16)*rms_y/w / mean_act, gamma={gamma}, dims |w|>=5, rms_y={rmsy:.3f} (estimate from mean act). Texts tf_screen_texts.json, hidden 16,20. Purpose: calibrate write size; score dLP(C)-dLP(W), dLP(R) vs base_lp7.npy.\n')
edited()
```