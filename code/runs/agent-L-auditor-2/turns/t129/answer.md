I have the construction. The mapping is simple: the target is divided by the post-MLP gain in each dimension, scaled by the layer's RMS, and divided by the neuron's activation, keeping dimensions with gain magnitude of at least 5. I'll use the same formula with the solve direction, gamma +0.3, and the register dimension masked out, then check the numbers before spending the pass. The write should be about 400, a similar size to the mode push of 444. I'm also creating the new DEV block and confirming it has no overlap with HOLDOUT.

```python
import numpy as np, json, torch
for x in list(edited()): drop_edit(x)
HP=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); MP=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
lay=15; j=4957
name=f'model.layers.{lay}.mlp.down_proj.weight'
Dm=load_original(name).float()
wg=load_original(f'model.layers.{lay}.post_feedforward_layernorm.weight').float().numpy()
mp=np.asarray(MP[:,lay],dtype=np.float32).mean(0); yp=Dm.numpy()@mp; rmsy=float(np.sqrt((yp**2).mean())); a=float(mp[j])
sd=np.load('/work/solve_dir_h16.npy')
def steer(gamma,tvec=sd):
    mask=np.abs(wg)>=5; mask[1038]=False
    t=np.zeros(1152,dtype=np.float32); t[mask]=gamma*tvec[mask]*rmsy/wg[mask]
    Dn=Dm.clone(); Dn[:,j]+=torch.tensor(t/a); save_edit(name,Dn)
    print('gamma',gamma,'kept dims',int(mask.sum()),'share kept',round(float(np.linalg.norm(tvec[mask])/np.linalg.norm(np.delete(tvec,1038))),3),'target write',round(float(np.linalg.norm(gamma*tvec[mask])),1),'rmsy',round(rmsy,4),'a',round(a,3),'col norm',round(float(Dm[:,j].norm()),3),'col change',round(float(np.linalg.norm(t/a)),3))
steer(0.3)
dev=json.load(open('/work/dev_ids.json')); ho=set(json.load(open('/work/holdout_ids.json')))
blk=dev[1024:1280]; assert len(blk)==256 and not set(blk)&ho
json.dump(blk,open('/work/dev_256e.json','w'))
open('/work/notes.md','a').write("""
Runs 99-104: neuron similarity scan: duplicate groups exist (L25 all3 54, L10 57, L11 15, L16 11). L10-16 carry 1-3% of write, negligible. L25 group = near-constant bias units (cv .13, act up to 7.9, 2.4 of 4.4 mean write, cos .88): static logit bias, explains 'repetition brake' of passes 4-5 (removal wiped static bias). No coherent write group otherwise: CIRCUIT-WITH-SHARED-WRITE CLASS EXCLUDED at cos>.5. BOS takes 16-50% attention in nearly all heads (attention sink, probably natural, may stabilise RMS); question sentence 1-5%.
Re-score token classes (R): operand digit baseline -.578, result -.708, operator -1.60, other -2.24; no edit has real positive slope on operand digits (best L17 x.5 +.0038).
SOLVE TASK-SET: mean last-prompt h16 solve minus summarise on 60 matched problems (/work/solve_dir_h16.npy): split cos .997, norm 1335 (between-problem 408), cos with style mode .165. Not difficulty confounded.
Pass 24 (both, /work/dev_256e.json = DEV[1024:1280], fresh): EDIT = model.layers.15.mlp.down_proj.weight col 4957 += gamma*t*rms_y/w/a, t = solve dir, gamma=+0.3, dims |w|>=5 except 1038, rms_y and a from gen_1 last prompt. Purpose: does stronger solving task-set raise accuracy? Next: gamma -0.3 on same block. Confirm: + up, - down. Refute: both flat or both down.
""")
```