That failure was my own: `F` got overwritten with a list after block 25, so `getH` built a bad path. The neuron table did print, and one row stands out: layer 20, neuron 1990 averages 2.11 at the last prompt token against -0.08 across answer tokens, far larger than its neighbours. It may just be a natural start-of-response detector, but it's worth keeping in mind given that layer 11's heads read that same token.

I'll fix the path by setting `getH` to the pass 4 folder explicitly, then rerun the volume comparison. In the same run I'll scan the MLP data from pass 2 for other neurons that are unusually active at the last prompt token, so I can see whether neuron 1990 is unique or one of many. The criteria for the volume idea are unchanged: a consistent eval-over-raw difference in signal share would keep it alive, and near-equality would close it.

```python
import os
F='/work/fwd_4/'
def getH(l): return np.load('/work/fwd_4/'+f'hidden_{l}.npy',mmap_mode='r')
rows_v=[]
ri = res_idx[:1500] if len(res_idx)>1500 else res_idx
oi = oth_s
def st(h):
    tot=(h**2).sum(1); rest=tot-(h[:,1038]**2)-(h[:,1125]**2)
    return h[:,1038], h[:,1125], np.sqrt(rest/1150), (h[:,1038]**2)/tot
for l in sorted(set(list(range(0,27,3))+[11,12,13,24,25,26])):
    H_=getH(l)
    for nm,idxs_ in [('res',ri),('oth',oi)]:
        r=PT.loc[idxs_]
        he_=np.array([H_[p*2,t-1] for p,t in zip(r.pi,r.te)],dtype=np.float32)
        hr_=np.array([H_[p*2+1,t-1] for p,t in zip(r.pi,r.tr)],dtype=np.float32)
        a1,b1,c1_,s1=st(he_); a2,b2,c2_,s2=st(hr_)
        rows_v.append(dict(l=l,tok=nm,c1038_e=a1.mean(),c1038_r=a2.mean(),sd1038_e=a1.std(),c1125_e=b1.mean(),c1125_r=b2.mean(),restRMS_e=c1_.mean(),restRMS_r=c2_.mean(),sig_ratio=(c1_/np.abs(a1)).mean()/(c2_/np.abs(a2)).mean(),share_e=s1.mean(),share_r=s2.mean()))
V_=pd.DataFrame(rows_v).sort_values(['tok','l'])
print(V_.round(3).to_string(index=False))

print(sorted(os.listdir('/work/gen_2_acts'))[:14], len(os.listdir('/work/gen_2_acts')))
print('ma/mp available:', 'ma' in globals(), 'mp' in globals(), (ma.shape, mp.shape) if 'ma' in globals() else None)
```