The reading is clear: where chat loses confidence, the probability moves to wrong digits (mean 0.47 vs. 0.06 in raw) and not to formatting tokens, and the top prediction is a digit in 96% of those cases. So the computation itself is disturbed. It isn't a style preference. Now for the depth profile. For each layer I'll compare the same predictive positions in eval and raw, masking scale coordinates 1038 and 1125 and normalising each state so the volume-control effect doesn't hide everything. I'll compute the size of the paired shift on result positions and on other positions, and how strongly the top components of that paired shift correlate with the per-token deficit. Because each token is its own control, this avoids confounding by problem difficulty. A correlation that appears at a clear depth would mark where the disturbance enters. If only the final layers correlate, I'm probably just seeing the output decision being read out, and I'd need a different contrast.

```python
mask=np.ones(1152,bool); mask[[1038,1125]]=False
def getH(l): return np.load(F+f'hidden_{l}.npy',mmap_mode='r')
res_idx=PT.index[PT.res].values; oth_idx=PT.index[~PT.digit].values
rng_=np.random.default_rng(0); oth_s=rng_.choice(oth_idx,size=min(1500,len(oth_idx)),replace=False)
def deltas(l,idx):
    H=getH(l); r=PT.loc[idx]
    he=np.array([H[p*2,t-1] for p,t in zip(r.pi,r.te)],dtype=np.float32)[:,mask]
    hr=np.array([H[p*2+1,t-1] for p,t in zip(r.pi,r.tr)],dtype=np.float32)[:,mask]
    he/= (np.sqrt((he**2).mean(1,keepdims=True))+1e-6); hr/= (np.sqrt((hr**2).mean(1,keepdims=True))+1e-6)
    return he,hr
out=[];store={}
dfc=PT.loc[res_idx,'lpe'].values-PT.loc[res_idx,'lpr'].values
for l in range(27):
    he,hr=deltas(l,res_idx); D=he-hr
    he2,hr2=deltas(l,oth_s); D2=he2-hr2
    store[l]=D
    U,S,Vt=np.linalg.svd(D-D.mean(0),full_matrices=False)
    sc=(D-D.mean(0))@Vt[:6].T
    cc=[np.corrcoef(sc[:,i],dfc)[0,1] for i in range(6)]
    b_=np.argmax(np.abs(cc))
    out.append(dict(l=l,sz_res=np.sqrt((D**2).mean()),sz_oth=np.sqrt((D2**2).mean()),meanshift_res=np.sqrt((D.mean(0)**2).mean()),meanshift_oth=np.sqrt((D2.mean(0)**2).mean()),
      cos_res_oth=float(D.mean(0)@D2.mean(0)/(np.linalg.norm(D.mean(0))*np.linalg.norm(D2.mean(0))+1e-9)),
      pc1_var=S[0]**2/(S**2).sum(),best_pc=int(b_),best_r=cc[b_],r_size=np.corrcoef(np.sqrt((D**2).mean(1)),dfc)[0,1]))
print(pd.DataFrame(out).set_index('l').round(3).to_string())
```