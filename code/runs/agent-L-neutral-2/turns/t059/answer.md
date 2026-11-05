Neuron 1031 is the clear choice as the writer. Its mean activation is 2.03, about 12 times larger than the other stable neurons, so the write needs only a small column change. Its activation holds up in generated answers too (2.03 ± 0.29 across answer means), but it drops to about 1.34 at the last prompt token, so the write will be a third weaker there. That matters little, because the target is the whole answer.

Strength 3 means a write of about 160 in hidden units, around 18% of the usual layer 15 update and 2% of the hidden norm. That's large enough to matter without overpowering the layer. Applying the edit to summary vectors will show whether it delivers the planned on-axis shift before I use a pass. If it does, I'll log it and run it on the first 256 problems of dev_b, which weren't used to derive the direction. The prediction is that if this axis causally affects correctness, accuracy should rise by a visible amount. A shift near zero with many changed answers would mean the axis describes correct answers without causing them.

```python
def build_steer(alpha, d, l=15, n=1031, rmsu=0.103, clampmin=3.0):
    Dn=f'model.layers.{l}.mlp.down_proj.weight'; D0=load_original(Dn).clone()
    w=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight'); scale=1+w
    t=torch.tensor(d,dtype=torch.float32)
    du=t*alpha/scale.sign()/scale.abs().clamp(min=clampmin)*rmsu
    an=2.032 if n==1031 else float(np.asarray(Ma[:,l,n],dtype=np.float32).mean())
    Dnew=D0.clone(); Dnew[:,n]+=du/an
    return Dn,D0,Dnew,w
Dn,D0,Dnew,w=build_steer(3.0,d16)
print('delta col norm %.3f, col norm %.3f, tensor relative %.5f'%(float((Dnew-D0)[:,1031].norm()),float(D0[:,1031].norm()),float((Dnew-D0).norm()/D0.norm())))
save_edit(Dn,Dnew)
D1=load(Dn); print('edited',edited())
def upd(D,a): 
    u_=a@D.T; return (u_/u_.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt())*(1+w)
ax=torch.tensor(d16/np.linalg.norm(d16),dtype=torch.float32)
Vs=torch.tensor(V[:4000]); dr=upd(D1,Vs)-upd(D0,Vs); pr_=dr@ax
print('token-level on-axis mean %.1f sd %.1f target %.1f |dr| %.1f cos(mean dr,axis) %.3f'%(float(pr_.mean()),float(pr_.std()),3*np.linalg.norm(d16),float(dr.norm(dim=1).mean()),float(dr.mean(0)@ax/dr.mean(0).norm())))
devb=json.load(open('/work/dev_b.json')); print(len(devb))
p7=devb[:256]; json.dump(p7,open('/work/p7_ids.json','w'))
import shutil; os.makedirs('/work/cand_S1',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_S1/') for f in os.listdir('/work/edited')]
open('/work/notes.md','a').write("""
Rounding noise: orig pass1 vs orig pass2 identical 159/256, correctness differs 4. Attention share: L17h0 problem .35-.43 (no re-read deficit, CLOSED); specials 27-69% of attention (bos parking, abstain).
EDIT S1 (current): layer15 down_proj col 1031 += alpha*d16*rmsu/((1+w) clamp3)/2.032, alpha=3, d16 = stratified correct-wrong_finished answer-mean h16 (dev_a), rmsu .103. saved /work/cand_S1.
pass7 plan: both, /work/p7_ids.json (dev_b[:256]).
""")
```