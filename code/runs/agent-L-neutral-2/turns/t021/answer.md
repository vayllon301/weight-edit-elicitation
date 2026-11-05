The shortlist is excellent. Several neurons are tightly locked to the last prompt token: for example layer 14 neuron 2785 is 0.94 at that token with a coefficient of variation of only 0.08 across problems, and its answer mean is about 0. Its correlation with style is mild (-0.18), so it isn't simply part of the style decision itself. That makes it a good delivery channel.

Rather than spend a separate pass confirming per-token silence, I'll build the causal test now, because it can answer both questions at once. The edit adds the Let-minus-Here mean-difference vector, measured after layer 14, to column 2785 of the layer 14 down_proj, scaled by the neuron's activation and the layer's normalisation. Only that one neuron's output changes, so the mode should shift at the answer's first token. The test set will be 235 problems the model originally opens with "Here", plus 21 "Let" problems as controls. If the neuron really is silent later, the controls should come out nearly identical. If "Let mode" is what helps, the flipped problems should gain around 9 points. If flips happen without any gain, the style correlation was just selection, and I'll drop this line.

In this step I'll build the edit, check it on summary vectors, save it, and write the notes.

```python
l=14;n=2785;strength=1.5
Dn=f'model.layers.{l}.mlp.down_proj.weight'; D0=load_original(Dn).clone()
w=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight')
X=torch.tensor(np.array(mp[:,l],dtype=np.float32)); u=X@D0.T
rmsu=u.pow(2).mean(-1).sqrt()
tau=torch.tensor(np.array(hp[ly,l+1],dtype=np.float32).mean(0)-np.array(hp[~ly,l+1],dtype=np.float32).mean(0))
scale=(1+w); print('1+w: min|.| %.2f median %.1f'%(float(scale.abs().min()),float(scale.abs().median())),'frac small',float((scale.abs()<3).float().mean()))
du=tau*strength/scale.sign()/scale.abs().clamp(min=3.0)*float(rmsu.mean())
an=float(X[:,n].mean()); dc=du/an
print('delta col norm %.3f, orig col norm %.3f, a_n %.3f'%(float(dc.norm()),float(D0[:,n].norm()),an))
Dnew=D0.clone(); Dnew[:,n]+=dc
save_edit(Dn,Dnew)
D1=load(Dn); print('stored change norm',float((D1-D0).norm()),'relative to tensor',float((D1-D0).norm()/D0.norm()))
def upd(D,a):
    u=a@D.T; return (u/u.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt())*(1+w)
dr=upd(D1,X)-upd(D0,X)
axu=tau/tau.norm()
pr=dr@axu
print('summary-vector effect at last prompt: on-axis mean %.1f std %.1f (target %.1f); |dr| %.1f; cos(dr,axis) %.3f'%(float(pr.mean()),float(pr.std()),float(tau.norm()*strength),float(dr.norm(dim=1).mean()),float((dr.mean(0)@axu)/dr.mean(0).norm())))
ans=torch.tensor(np.array(ma[:,l],dtype=np.float32)); dra=upd(D1,ans)-upd(D0,ans)
print('same edit on answer-mean activation: |dr| %.1f'%float(dra.norm(dim=1).mean()))
here=[order[i] for i in range(512) if not ly[i]]; let=[order[i] for i in range(512) if ly[i]]
ids=here+let[:256-len(here)]; json.dump(ids,open('/work/p2_ids.json','w')); print(len(here),len(ids))
with open('/work/notes.md','a') as f: f.write(f"""
pass2 plan: EDIT A = layer14 down_proj column 2785 += tau*1.5/(1+w)*rms_u/a_n (tau=mean Let-Here h15 last prompt, dev_a). Causal test style->accuracy. Set /work/p2_ids.json: 235 orig-Here + {256-len(here)} orig-Let controls (all dev_a). both.
Findings so far: acc 19.3%, cut 27%, per-step survival .6; Let-Here +9.1 pts stratified; opener margin narrow (69% within 2 logits); arithmetic mostly OK; no extraction misses; hidden index 26 is pre-final-norm.
""")
print(edited())
```