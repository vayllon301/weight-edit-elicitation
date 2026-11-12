The earlier analysis used a label array `g` ('v0', 'v3', 'chat'), together with `starts`, `L` and `N`, all still in the kernel. I'll reuse them as they are and run the planned AND-selectivity scan over layers 13–25 of fwd_41 (E14). The scan uses per-text answer-token means (nan-aware, because padding is NaN) and scores each neuron by |v0 − nearest other group| / pooled sd.

```python
print(N, collections.Counter(g.tolist()), np.array_equal(np.array(L),ln41))
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