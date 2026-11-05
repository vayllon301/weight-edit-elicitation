My reused code matches, so I'm building the edit with the same machinery. Layer 15's #4957 is a strong choice, with a ratio of 11 between its last-token activation and its typical size elsewhere. At other tokens it writes only about 9% of the planned size, with a roughly zero mean.

The target is the mean L-minus-H last-prompt state at hidden 16, from the baseline answers. I'm starting at twice that gap (γ=2), which should carry a typical H-mode problem past the boundary if the direction is causal.

I'll run this on a fresh DEV block (DEV[512:768]). It wasn't used to build the direction and hasn't been used in any pass, so the accuracy estimate is less likely to be flattered by earlier choices. I'll record the edit and problem set in the notes first, as required.

```python
import numpy as np, json, torch
HP=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); MP=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
lay=15; hid=16; j=4957; gamma=2.0
X=np.asarray(HP[:,hid],dtype=np.float32)
tvec=X[isL].mean(0)-X[~isL].mean(0)
np.save('/work/modeLH_h16.npy',tvec)
name=f'model.layers.{lay}.mlp.down_proj.weight'
Dm=load_original(name).float()
w=load_original(f'model.layers.{lay}.post_feedforward_layernorm.weight').float().numpy()
mp=np.asarray(MP[:,lay],dtype=np.float32).mean(0)
yp=Dm.numpy()@mp; rmsy=float(np.sqrt((yp**2).mean())); a=float(mp[j])
mask=np.abs(w)>=5
t=np.zeros(1152,dtype=np.float32); t[mask]=gamma*tvec[mask]*rmsy/w[mask]
Dn=Dm.clone(); Dn[:,j]+=torch.tensor(t/a)
print('mode gap',round(float(np.linalg.norm(tvec)),1),'kept dims',int(mask.sum()),'share of gap kept',round(float(np.linalg.norm(tvec[mask])/np.linalg.norm(tvec)),3),'target',round(float(np.linalg.norm(gamma*tvec)),1),'rms_y(last prompt)',round(rmsy,4),'a',round(a,3))
print('col norm',round(float(Dm[:,j].norm()),3),'col change',round(float((Dn-Dm)[:,j].norm()),3),'tensor rel',round(float((Dn-Dm).norm()/Dm.norm()),4))
save_edit(name,Dn)
dev=json.load(open('/work/dev_ids.json'))
c=dev[512:768]; json.dump(c,open('/work/dev_256c.json','w')); print('dev_256c',len(c),'overlap with first 512',len(set(c)&set(dev[:512])),'overlap holdout',len(set(c)&set(json.load(open('/work/holdout_ids.json')))))
with open('/work/notes.md','a') as f:
    f.write(f"""
Closed run 52: outsized near-constant neurons (L23 #5826, L16 #3437, L10 #1868/768, L22 #5108/4909) are bias units: constant across positions, sparse columns, mirror pairs add rather than cancel. H first heading 'calculate the total' acc 13% vs other 24%, matched by k (k2 22/44, k3 3/15), may be selection.
Pass 11 (both, activations layers 13,16,20,26, /work/dev_256c.json = DEV[512:768], fresh): EDIT = {name} col {j} += gamma*t*rms_y/w/a, t = mean last-prompt h16 L minus H (baseline gen_1, /work/modeLH_h16.npy), gamma={gamma}, dims |w|>=5, rms_y={rmsy:.4f}, a={a:.3f}. Neuron fires last prompt token only (ratio 11). Purpose: causal test of mode. Confirm: H->L conversion, acc up. Refute: conversion but flat acc.
""")
edited()
```