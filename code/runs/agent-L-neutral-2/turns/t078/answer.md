Pooling tells a clear story, and it kills my subgroup hopes. The Here-style gain from pass 11 reversed on fresh problems, leaving Here at +0.4 ± 2.3 **even though its loops almost vanished** (17.2% to 1.3%). So loops were being removed without any gain. Non-flipped answers overall are at +0.5 ± 1.8, and the flipped group at -4.2 ± 5.1. Those flips hurt on the point estimate, but they aren't significant and they don't explain everything.

There's a more telling number here. The 97 answers that escaped gained 7 and lost 2, a net **5%** conversion. That's almost exactly what the *exit* test predicted (6 of 121, or 5%), and far below the *avoidance* estimate of 13.3%. Unused numbers also rose in both passes (20.6% to 25.7% and 16.4% to 23.9%), and escaped answers end at about the same length as the point where loops usually begin. All of this suggests a specific explanation: **L2 doesn't avoid loops, it makes the model stop early.** That would make sense of it, because the axis is in-loop minus finished, and finished answers contain the ending tokens, so reversing the axis may simply push towards ending.

That's testable. I'll measure the "ending direction" from pass 9 (final-line tokens minus interior tokens at hidden index 16) and check how much of it lies inside the cut axis. A strong negative alignment would confirm it, and it would also suggest a principled variant: remove that component, so the push works against repetition and not in favour of stopping, and use a neuron that's quiet at the prompt so the opener stays unchanged. If that variant keeps cuts low while coverage recovers, it's real avoidance, and accuracy should improve. If cuts return, loop removal only ever worked by stopping, and this line is finished.

```python
import numpy as np, json
d='/work/fwd_9/'
H9=np.load(d+'hidden_16.npy',mmap_mode='r'); ids9=np.load(d+'input_ids.npy',mmap_mode='r')
K=pd.read_pickle('/work/k9.pkl'); L9=json.load(open(d+'meta.json'))['lengths']
ints=[];ends=[];lab=[]
for i in range(len(K)):
    n=L9[i]; x=np.asarray(ids9[i,:n]); sot=[j for j,v in enumerate(x) if v==105]; a0=sot[1]+3
    p=int(K.p[i]); h=np.asarray(H9[i,a0:n],dtype=np.float32).astype(np.float64)
    ints.append(h[:p-a0+1].mean(0)); ends.append(h[p-a0+1:].mean(0)); lab.append(K.sty[i])
ints=np.stack(ints); ends=np.stack(ends); lab=np.array(lab)
end_dir=np.mean([ends[lab==s_].mean(0)-ints[lab==s_].mean(0) for s_ in ('Let','Here')],0)
print('|end_dir| %.1f cos(cut_dir,end_dir) %.3f cos(d16,end_dir) %.3f'%(np.linalg.norm(end_dir),cos(cut_dir,end_dir),cos(d16,end_dir)))
# interior-only comparison: cut answer means vs finished interior means
Xcut=X16[cut]; 
ci=[]
for s_,lb in ((0,'Let'),(1,'Here')):
    ci.append(X16[cut&(here==s_)].mean(0)-ints[lab==lb].mean(0))
cut_int=np.mean(ci,0); print('|cut_int| %.1f cos(cut_int,cut_dir) %.3f cos(cut_int,end_dir) %.3f'%(np.linalg.norm(cut_int),cos(cut_int,cut_dir),cos(cut_int,end_dir)))
e_=end_dir/np.linalg.norm(end_dir)
cut_perp=cut_dir-(cut_dir@e_)*e_; print('|cut_perp| %.1f share removed %.3f'%(np.linalg.norm(cut_perp),1-(np.linalg.norm(cut_perp)/np.linalg.norm(cut_dir))**2))
# answer-only writer candidates at layer 15
mA15=np.asarray(Ma[:,15],dtype=np.float32); mP15=np.asarray(Mp[:,15],dtype=np.float32)
ma=mA15.mean(0); sa=mA15.std(0); mp_=mP15.mean(0)
A15=np.load('/work/fwd_4/mlp_act_15.npy',mmap_mode='r')
sc_=np.abs(ma)/(sa+1e-3)
ok=np.where((np.abs(ma)>0.25)&(np.abs(mp_)<0.25*np.abs(ma)))[0]
top=ok[np.argsort(-sc_[ok])[:8]]
print('n cand',len(ok))
for n_ in top:
    print(int(n_),'ans mean %.3f sd %.3f | last prompt %.3f | token sd(pass4) %.3f token mean %.3f | col %.3f'%(ma[n_],sa[n_],mp_[n_],np.nanstd(V[:,n_]),np.nanmean(V[:,n_]),float(D0[:,n_].norm())))
```