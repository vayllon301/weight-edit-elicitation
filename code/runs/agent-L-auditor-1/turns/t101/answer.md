Now I'll compute the same profile for T3 next to the original and T2, along with the scores. I'll also redo the alignment of what remains at L12 against the candidate writes, including `du` through head 1 and how much of the remaining shift lies along what T3 removed. That last number again indicates whether strength has room.

```python
import numpy as np, pandas as pd, torch
rows=[]
for l in range(10,27):
    r={'l':l}
    for tag,F in [('t3','/work/fwd_22/')]:
        he,hr=deltas2(F,l,res_idx); D=he-hr
        he2,hr2=deltas2(F,l,oth_s); D2=he2-hr2
        ms[(tag,l)]=(D.mean(0),D2.mean(0))
        r['o_res']=df.loc[l,'o_res']; r['t2_res']=df.loc[l,'t_res']; r['t3_res']=np.sqrt((D.mean(0)**2).mean())
        r['o_oth']=df.loc[l,'o_oth']; r['t2_oth']=df.loc[l,'t_oth']; r['t3_oth']=np.sqrt((D2.mean(0)**2).mean())
        r['o_sz']=df.loc[l,'o_sz']; r['t2_sz']=df.loc[l,'t_sz']; r['t3_sz']=np.sqrt((D**2).mean())
    rows.append(r)
d3=pd.DataFrame(rows).set_index('l')
d3['t3_inc']=[np.nan]+[float(np.sqrt(((ms[('t3',l)][1]-ms[('t3',l-1)][1])**2).mean())) for l in range(11,27)]
print(d3.round(3).to_string())
print(score_pair('/work/fwd_22').to_string())
he_o,hr_o=deltas2('/work/fwd_4/',12,oth_s); he_t,hr_t=deltas2('/work/fwd_22/',12,oth_s)
r_e=(he_t-he_o).mean(0); s_o=(he_o-hr_o).mean(0); s_t=(he_t-hr_t).mean(0)
print('cos(removed_e,shift_o) %.3f proj coef remaining on removed %.3f cos %.3f |shift| %.1f->%.1f'%(cs(r_e,s_o),pr(s_t,r_e),cs(s_t,r_e),np.linalg.norm(s_o),np.linalg.norm(s_t)))
inc=(ms[('t3',12)][1]-ms[('o',11)][1])
du=torch.load('/work/du_anchor_diff.pt').float(); du=du/du.norm()
n_='model.layers.11.self_attn.o_proj.weight'; Oc=load(n_).float(); O0=load_original(n_).float()
print('remaining inc rms %.3f'%float(np.sqrt((inc**2).mean())))
for h in range(4):
    w=(Gt*(O0[:,h*256:(h+1)*256]@du)).numpy()[mask]
    Oh=(Gt[:,None]*Oc[:,h*256:(h+1)*256]).numpy()[mask]
    U,S,_=np.linalg.svd(Oh,full_matrices=False); k=(S>S[0]*1e-3).sum(); Q=U[:,:k]
    f=lambda x: float(((Q.T@x)**2).sum()/(x@x))
    print(f'h{h} orig du write cos inc {cs(w,inc):+.3f}; writable share inc {f(inc):.3f}')
Va=torch.tensor(res[11]['V']); am=torch.tensor(res[11]['ans'].mean(0))
names=['bos','sot1','user','nl_u','eot','nl_e','sot2','model','nl_m']
for j,nm in enumerate(names):
    v=Va[:,j].mean(0)-am
    print(nm,'|v-ans| %.1f cos(du) %+.3f'%(float(v.norm()),float(v@du/v.norm())),' cos write->inc by head',[round(cs((Gt*(Oc[:,h*256:(h+1)*256]@v)).numpy()[mask],inc),2) for h in range(4)])
```