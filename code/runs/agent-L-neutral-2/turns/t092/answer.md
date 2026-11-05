Measured on answer tokens only, the neurons are fairly selective but no single one is ideal. n311 is stable across problems (CV 0.13) but leaks more (7.9% of tokens above a quarter of its prompt activation). n1409 leaks little (1.2%) but is variable (CV 0.26).

Splitting the write across several neurons should fix both. All of them fire at the last prompt token, so their shares add up there, while their answer-token leaks are less aligned and should partly cancel. Averaging also reduces the variability between problems.

I'll build this for 'First', the reference-style opener with the plainest continuation, using five writers: n1409, n2167, n4081, n5528 and n311. The strength will be set from the weights to give about 15 logits, against a median gap of 10.9. I'll check it on the mean last-prompt state only. That check covers whether 'First' becomes the top token, whether the formula reproduces the known gap, and how far the leak falls.

If 'First' wins and combined leak is low, I'll run a paired pass. If some other token wins, or the leak stays high, I'll fix the build before spending a pass. The test itself asks whether a first word the model never naturally chooses leads to better reasoning.

```python
import numpy as np, torch, os, shutil, re
cand=[1409,2167,4081,5528,311]
rat=O[:,cand]/mp[cand]; comb=rat.mean(1); pr=(p[:,cand]/mp[cand]).mean(1)
print('combined leak rms %.3f p99 %.3f max %.3f frac>.25 %.4f | prompt activation ratio mean %.3f cv %.3f'%(np.sqrt((comb**2).mean()),np.quantile(np.abs(comb),.99),np.abs(comb).max(),(np.abs(comb)>.25).mean(),pr.mean(),pr.std()/pr.mean()))
Dn='model.layers.25.mlp.down_proj.weight'; D0=load_original(Dn)
uP=torch.tensor(p)@D0.T; rmsP=uP.pow(2).mean(1).sqrt(); uA=torch.tensor(O)@D0.T; rmsA=uA.pow(2).mean(1).sqrt()
print('rms u: prompt mean %.4f cv %.2f; answer median %.4f p1 %.4f p99 %.4f'%(float(rmsP.mean()),float(rmsP.std()/rmsP.mean()),float(rmsA.median()),float(rmsA.quantile(.01)),float(rmsA.quantile(.99))))
HL=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); print('HL',HL.shape)
hbar=torch.tensor(np.asarray(HL[:,26]).astype(np.float32).mean(0)); rms=lambda x:x.pow(2).mean().sqrt()
ids_by={tok.decode([a]):a for a in pm}; idL,idH=ids_by['Let'],ids_by['Here']
E=load('model.embed_tokens.weight'); s_=1+load('model.norm.weight'); sc=1+load_original('model.layers.25.post_feedforward_layernorm.weight')
rmsu_p=float(rmsP.mean()); rmsh=float(rms(hbar)); abar=torch.tensor(p.mean(0))
def build_opener(name,delta,neurons=cand,clampfrac=.05):
    tid=ids_by[name]; g=s_*(E[tid]-.5*(E[idL]+E[idH])); v=g*(delta*rmsh)/(g@g)
    cl=sc.abs().median()*clampfrac; du=v/sc.sign()/sc.abs().clamp(min=cl)*rmsu_p
    Dnew=D0.clone()
    for n_ in neurons: Dnew[:,n_]+=du/(len(neurons)*float(mp[n_]))
    return Dnew,v
def check(Dnew,name):
    tid=ids_by[name]; u0=D0@abar; u1=Dnew@abar; o0=sc*u0/rms(u0); o1=sc*u1/rms(u1); h1=hbar+o1-o0
    l0=E@(s_*hbar/rms(hbar)); l1=E@(s_*h1/rms(h1))
    f=lambda l:{k:round(float(l[i]),2) for k,i in [('Let',idL),('Here',idH),(name,tid)]}
    top=torch.topk(l1,4).indices.tolist(); safe=[tok.decode([i]) if re.fullmatch(r"[\sA-Za-z'*#$0-9.:-]{0,12}",tok.decode([i])) else '<other>' for i in top]
    print('before',f(l0),'\nafter ',f(l1),'\ntop after',safe,'| |o1-o0| %.1f |o0| %.1f rel u change %.3f'%(float((o1-o0).norm()),float(o0.norm()),float((u1-u0).norm()/u0.norm())))
Dnew,v=build_opener('First',15.0); print('|v| %.1f rmsh %.1f g-norm %.1f'%(float(v.norm()),rmsh,float((s_*(E[ids_by['First']]-.5*(E[idL]+E[idH]))).norm())))
check(Dnew,'First')
print('col delta max %.3f, tensor rel change %.5f'%(max(float((Dnew-D0)[:,n_].norm()) for n_ in cand),float((Dnew-D0).norm()/D0.norm())))
```