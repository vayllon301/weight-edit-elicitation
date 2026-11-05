That's a minor slip: the pickle line came after the crash in the earlier run, so the file was never written. `K` itself is still in the kernel, so I'll save it now and run the same analysis from memory.

```python
K.to_pickle('/work/k9.pkl')
d='/work/fwd_9/'
H9=np.load(d+'hidden_16.npy',mmap_mode='r'); ids9=np.load(d+'input_ids.npy',mmap_mode='r')
L9=json.load(open(d+'meta.json'))['lengths']
ints=[];ends=[];lab=[]
for i in range(len(K)):
    n=L9[i]; x=np.asarray(ids9[i,:n]); sot=[j for j,v in enumerate(x) if v==105]; a0=sot[1]+3
    p=int(K.p[i]); h=np.asarray(H9[i,a0:n],dtype=np.float32).astype(np.float64)
    ints.append(h[:p-a0+1].mean(0)); ends.append(h[p-a0+1:].mean(0)); lab.append(K.sty[i])
ints=np.stack(ints); ends=np.stack(ends); lab=np.array(lab)
end_dir=np.mean([ends[lab==s_].mean(0)-ints[lab==s_].mean(0) for s_ in ('Let','Here')],0)
print('|end_dir| %.1f cos(cut_dir,end_dir) %.3f cos(d16,end_dir) %.3f'%(np.linalg.norm(end_dir),cos(cut_dir,end_dir),cos(d16,end_dir)))
ci=[]
for s_,lb in ((0,'Let'),(1,'Here')):
    ci.append(X16[cut&(here==s_)].mean(0)-ints[lab==lb].mean(0))
cut_int=np.mean(ci,0); print('|cut_int| %.1f cos(cut_int,cut_dir) %.3f cos(cut_int,end_dir) %.3f'%(np.linalg.norm(cut_int),cos(cut_int,cut_dir),cos(cut_int,end_dir)))
e_=end_dir/np.linalg.norm(end_dir)
cut_perp=cut_dir-(cut_dir@e_)*e_; print('|cut_perp| %.1f share of cut_dir energy removed %.3f'%(np.linalg.norm(cut_perp),1-(np.linalg.norm(cut_perp)/np.linalg.norm(cut_dir))**2))
mA15=np.asarray(Ma[:,15],dtype=np.float32); mP15=np.asarray(Mp[:,15],dtype=np.float32)
ma=mA15.mean(0); sa=mA15.std(0); mp_=mP15.mean(0)
sc_=np.abs(ma)/(sa+1e-3)
ok=np.where((np.abs(ma)>0.25)&(np.abs(mp_)<0.25*np.abs(ma)))[0]
top=ok[np.argsort(-sc_[ok])[:8]]
print('n cand',len(ok))
for n_ in top:
    print(int(n_),'ans mean %.3f sd %.3f | last prompt %.3f | token sd(pass4) %.3f token mean %.3f | col %.3f'%(ma[n_],sa[n_],mp_[n_],np.nanstd(V[:,n_]),np.nanmean(V[:,n_]),float(D0[:,n_].norm())))
```