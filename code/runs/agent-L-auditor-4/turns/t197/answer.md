The label array `g` is intact (40 v0, 40 v3, 24 chat). But `L` was overwritten by a scalar in block 146, during the attention segmentation. fwd_41 is on the same tf10 texts, so its own per-text lengths, ln41, are the correct ones. I'll restore `L = ln41` and rerun the same scan without other changes.

```python
L=ln41
res=[]
for l in range(13,26):
    X=np.load(f'/work/fwd_41/mlp_act_{l}.npy',mmap_mode='r')
    M=np.stack([np.nanmean(X[i,starts[i]:L[i]].astype(np.float32),0) for i in range(N)])
    G={k:M[g==k] for k in ['v0','v3','chat']}
    sd=np.sqrt(sum(G[k].var(0) for k in G)/3)+1e-3
    m0,m3,mc=(G[k].mean(0) for k in ['v0','v3','chat'])
    near=np.where(np.abs(m3-m0)<np.abs(mc-m0),m3,mc)
    s=np.abs(m0-near)/sd
    for j in np.argsort(-s)[:3]: res.append((l,int(j),round(float(s[j]),2),round(float(m0[j]),2),round(float(m3[j]),2),round(float(mc[j]),2)))
sorted(res,key=lambda r:-r[2])[:15]
```