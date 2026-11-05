The neuron-level lead is closed. L11 looks just like its neighbours on dispersion (1.44 against 1.23 to 1.41), top-1% share (0.031, the same) and gate-up agreement by neuron (0.05, the same). The top neurons are only 2 to 3 times their expected count, with ordinary norms, so there's no moved module to remove.

One number does differ, though. The correlation between `gate_proj` and `up_proj` across **input columns** is 0.53 in L11, against about 0.2 to 0.25 in the neighbouring layers. So movement in L11's MLP may focus on particular stream dimensions rather than particular neurons. It might also be an artefact, because columns with larger weights have more eligible entries in both tensors, which alone would produce a correlation.

I'll normalise by the eligible count and see whether a few columns truly stand out. If they do, I'll check whether those dimensions overlap the output of h3's tail, which would link L11's attention write to its own MLP read. If the excess disappears after normalising, the lead is closed. Either way I'll prepare the next dose test, so this is the last forensic step before the remaining passes.

```python
import numpy as np, torch
out={}
for L in (10,11,13):
    mg,bg=offmap(f'model.layers.{L}.mlp.gate_proj.weight',121); mu,bu=offmap(f'model.layers.{L}.mlp.up_proj.weight',121)
    cg=mg.sum(0).astype(float); cu=mu.sum(0).astype(float); bcg=bg.sum(0).astype(float); bcu=bu.sum(0).astype(float)
    rg=cg/np.maximum(bcg,1); ru=cu/np.maximum(bcu,1)
    ok=(bcg>=200)&(bcu>=200)
    rr=float(np.corrcoef(rg[ok],ru[ok])[0,1])
    tot=(cg+cu)/np.maximum(bcg+bcu,1); rate=(cg.sum()+cu.sum())/(bcg.sum()+bcu.sum())
    exp=(bcg+bcu)*rate; z=(cg+cu-exp)/np.sqrt(np.maximum(exp,1))
    out[L]=(z,ok,bcg+bcu)
    print('L',L,'cols ok',int(ok.sum()),'r(rate gate,up)',round(rr,3),'r(big gate,up)',round(float(np.corrcoef(bcg,bcu)[0,1]),3),'z sd',round(float(z[ok].std()),2),'n z>6',int((z[ok]>6).sum()),'max z',round(float(z[ok].max()),1))
z,ok,bb=out[11]
idx=np.argsort(-np.where(ok,z,-99))[:14]
pre=load_original('model.layers.11.pre_feedforward_layernorm.weight').float()
post=load_original('model.layers.11.post_attention_layernorm.weight').float()
print('top L11 cols (dim z big preGain postAttnGain)',[(int(i),round(float(z[i]),1),int(bb[i]),round(float(1+pre[i]),2),round(float(1+post[i]),2)) for i in idx])
print('median |pre gain|',round(float((1+pre).abs().median()),2),'median |post gain|',round(float((1+post).abs().median()),2))
Bt=O11[:,768:1024]; Ut_,St_,Vt_=torch.linalg.svd(tail11,full_matrices=False)
row=(tail11**2).sum(1).numpy(); r_all=row/row.mean()
print('tail row energy at top cols /mean',[round(float(r_all[i]),2) for i in idx[:14]])
hi=np.where(ok&(z>6))[0]
print('n hi',len(hi),'mean tail row energy ratio hi',round(float(r_all[hi].mean()) if len(hi) else 0,2),'rest',round(float(r_all[ok&~(z>6)].mean()),2),'corr z vs tail energy',round(float(np.corrcoef(z[ok],r_all[ok])[0,1]),3))
```