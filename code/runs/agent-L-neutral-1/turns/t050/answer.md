**What the last result showed.** Selectivity varies a lot, and the token-level check was worth doing. L17 n4587, my earlier favourite, fails: it reaches 5.06 elsewhere against 2.41 at the target.

Two units pass:
- **L23 n1446:** activation 2.57 at the target, RMS 0.07 elsewhere (2.7% of target). Its only notable leak is the blank line before the instruction, about 30% of target at that one position.
- **L11 n4788:** activation -1.42, CV 0.06, RMS 0.08 elsewhere (5.8%), with leaks spread over more tokens.

**What I conclude.** L23 n1446 is the cleanest writer, and it acts late enough that the effect on the opening word should be direct. Its CV of 0.24 means the push will vary by problem, but its sign is consistent.

**What I will do now.** I'll build the candidate by hand from statistics: the mean last-prompt residual difference between "Let" and "Here's" answers, excluding x1038, written through column 1446 of `L23.down_proj`. First I need to verify three assumptions on summary vectors:
- `hidden l` means after block l-1. The predicted MLP output of L23 should align better with h24-h23 than with h25-h24.
- The scaling that goes through the post-FF RMSNorm.
- Survival of the small column change after bfloat16 rounding.

**What would confirm or refute it.** If all three checks pass, I'll save the edit at full strength and log it, ready for a causal "both" test on the unused DEV chunk. If the layer mapping is off, or rounding destroys the write, I'll fix that first.

```python
nmD='model.layers.23.mlp.down_proj.weight'; nmW='model.layers.23.post_feedforward_layernorm.weight'
print([n for n in NAMES if 'layers.23.' in n][:20])
D=load(nmD).float(); w=load(nmW).float(); print('D',D.shape,'w med',round(float(w.median()),2),'min',round(float(w.min()),2),'share |1+w|<5',round(float(((1+w).abs()<5).float().mean()),4))
if D.shape[0]!=1152: print('unexpected orientation')
D_=D.numpy(); w_=w.numpy()
Mp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r'); P=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
j23=layers_m.index(23); m_=np.array(Mp[:,j23,:],dtype=np.float32)
h23=np.array(P[:,hl.index(23),:],dtype=np.float32); h24=np.array(P[:,hl.index(24),:],dtype=np.float32); h25=np.array(P[:,hl.index(25),:],dtype=np.float32)
mbar=m_.mean(0); out=D_@mbar; rms=float(np.sqrt((out**2).mean())); Pf=out/rms*(1+w_)
mask=np.ones(1152,bool); mask[1038]=False
cos=lambda a,b:float(a[mask]@b[mask]/(np.linalg.norm(a[mask])*np.linalg.norm(b[mask])+1e-9))
print('cos(predicted L23 postFF, h24-h23)',round(cos(Pf,(h24-h23).mean(0)),3),'cos(.,h25-h24)',round(cos(Pf,(h25-h24).mean(0)),3))
print('proj coef on h24-h23',round(float(Pf[mask]@(h24-h23).mean(0)[mask]/(Pf[mask]@Pf[mask])),3))
out_s=m_[:64]@D_.T; rms_s=np.sqrt((out_s**2).mean(1)); print('rms(out) per sample mean/sd',round(float(rms_s.mean()),4),round(float(rms_s.std()),4),'rms of mean',round(rms,4))
md=df['mode'].values
d24=h24[md=='let'].mean(0)-h24[md=='heres'].mean(0); d24[1038]=0
print('|d24|',round(float(np.linalg.norm(d24)),1),'top coord share',round(float((d24**2).max()/(d24**2).sum()),4),'|h24 rest| typical',round(float(np.linalg.norm(h24[:, mask],axis=1).mean()),1))
n_=1446; a_=float(mbar[n_]); print('a',round(a_,3))
scale=(1+w_).copy(); floor=0.1*np.median(np.abs(scale)); good=np.abs(scale)>=floor
print('coords dropped by floor',int((~good).sum()),'share of d24 energy there',round(float((d24[~good]**2).sum()/(d24**2).sum()),4))
def col_delta(alpha,rms_use):
    k=np.zeros(1152,np.float32); k[good]=alpha*d24[good]*rms_use/(a_*scale[good]); return k
k1=col_delta(1.0,float(rms_s.mean()))
# effect on a summary vector, including the renormalisation by the new rms
def effect(k):
    o2=out+a_*k; r2=float(np.sqrt((o2**2).mean())); dh=o2/r2*(1+w_)-out/rms*(1+w_)
    return dh,r2
dh,r2=effect(k1)
print('rms before/after',round(rms,4),round(r2,4),'cos(dh,d24)',round(cos(dh,d24),3),'gain along d24',round(float(dh[mask]@d24[mask]/(d24[mask]@d24[mask])),3),'dh x1038',round(float(dh[1038]),1))
cn=D_[:,n_]; print('|col|',round(float(np.linalg.norm(cn)),3),'|k|',round(float(np.linalg.norm(k1)),3),'ratio',round(float(np.linalg.norm(k1)/np.linalg.norm(cn)),2),'median abs col',round(float(np.median(np.abs(cn))),5),'median abs k',round(float(np.median(np.abs(k1))),5))
import torch
E=D.clone(); E[:,n_]=E[:,n_]+torch.tensor(k1); save_edit(nmD,E)
E2=load(nmD).float(); got=(E2-D)[:,n_].numpy(); oth=(E2-D).clone(); oth[:,n_]=0
print('after rounding: cos(got,k)',round(float(got@k1/(np.linalg.norm(got)*np.linalg.norm(k1))),5),'gain',round(float(got@k1/(k1@k1)),4),'other cols changed',int((oth!=0).sum()),'edited',edited())
```