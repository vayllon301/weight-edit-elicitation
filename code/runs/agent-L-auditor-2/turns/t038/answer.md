The pairing check came back clean, and that's useful. Gate/up/down are clearly coupled in every layer, with spreads 3–8 times the shuffled null, so there's no wholesale scrambling. No neurons are dead (minimum norms are all reasonable), and kurtosis shows no break at layers 23–25, so there's no sign of broad post-hoc changes there.

One signal remains. In layers 19–24, a set of neurons has `down_proj` columns far larger than their `up_proj` row would predict, with z up to 27 and 21–46 neurons per layer above z = 6. Some of that is probably inflated, because the main group in those layers is so tight that a robust z-score overstates any outlier. Still, it's exactly the shape a "strong writer" would have.

So I'll characterise them. Are they near-constant units, and what share of the layer's output do they carry? Do their columns share a direction within and across layers, which would suggest a coordinated circuit, or do they point in unrelated directions? A shared direction with a notable output share would be a real target, and I'd test it by knocking those neurons out. Unrelated directions with small shares would make them just the natural tail, and I'd close this line.

```python
import torch, numpy as np, pandas as pd
mma=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
rows_=[]; cand={}; top_dirs={}
for l_ in range(17,26):
    U=load_original(f'model.layers.{l_}.mlp.up_proj.weight').float()
    D=load_original(f'model.layers.{l_}.mlp.down_proj.weight').float()
    un=U.norm(dim=1).log().numpy(); dn=D.norm(dim=0).log().numpy()
    b=np.polyfit(un,dn,1); res=dn-np.polyval(b,un); mad=np.median(np.abs(res-np.median(res)))*1.4826
    z=(res-np.median(res))/mad
    S=np.where(z>6)[0]
    cand[l_]=S
    m=np.asarray(mma[:,l_,:],dtype=np.float32); a=m.mean(0); sd=m.std(0)
    cv=np.abs(sd/(np.abs(a)+1e-6))
    Dn=D.numpy()
    wr=np.abs(a)*np.linalg.norm(Dn,axis=0)       # mean write of each neuron
    tot=(Dn*a[None,:]).sum(1)                    # net mean write of the layer
    if len(S)==0: continue
    sub=(Dn[:,S]*a[S][None,:]).sum(1)
    C=Dn[:,S]/np.linalg.norm(Dn[:,S],axis=0,keepdims=True)
    sv=np.linalg.svd(C,compute_uv=False); 
    gram=C.T@C; off=gram[~np.eye(len(S),dtype=bool)]
    U_,_,_=np.linalg.svd(Dn[:,S],full_matrices=False); top_dirs[l_]=U_[:,0]
    rows_.append(dict(l=l_,n=len(S),z_top=round(float(z.max()),1),d_ratio=round(float(np.exp(np.median(dn[S])-np.median(dn))),2),
       acts_abs=round(float(np.median(np.abs(a[S]))),3),acts_all=round(float(np.median(np.abs(a))),3),cv=round(float(np.median(cv[S])),2),cv_all=round(float(np.median(cv)),2),
       share_sumwrite=round(float(wr[S].sum()/wr.sum()),3),netwrite_ratio=round(float(np.linalg.norm(sub)/np.linalg.norm(tot)),2),cos_net=round(float(sub@tot/np.linalg.norm(sub)/np.linalg.norm(tot)),2),
       sv1_share=round(float(sv[0]**2/(sv**2).sum()),2),offcos_abs=round(float(np.abs(off).mean()),3),offcos_max=round(float(np.abs(off).max()),2),
       top3=[(int(i),round(float(a[i]),2)) for i in S[np.argsort(-z[S])[:3]]]))
print(pd.DataFrame(rows_).to_string(index=False))
ls=sorted(top_dirs)
M=np.array([[abs(float(top_dirs[a]@top_dirs[b])) for b in ls] for a in ls])
print('layers',ls); print(np.round(M,2))
```