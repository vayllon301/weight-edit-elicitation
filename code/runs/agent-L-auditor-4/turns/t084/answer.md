The tf6 metadata gives each text's (expression id, op, context, result), and tf8_sel maps the pass-13 texts into it. I'll use this to stratify error rates and the wrong/right separation by operation. I'll also switch "wrong" to the stricter, exact-match definition: any result digit not top-1.

```python
tk=np.load(d+'topk_ids.npy',mmap_mode='r')
def exact(i):
    n,k=positions(i); return all(int(tk[i,j-1,0])==int(ids[i,j]) for j in range(n-1-k,n-1))
ops=np.array([m6[sel[i]][1] for i in A]); ctx=[m6[sel[i]][2] for i in A[:3]]+[m6[sel[i]][2] for i in B[:3]]
print(ctx)
exA=np.array([exact(i) for i in A]); exB=np.array([exact(i) for i in B])
for op in np.unique(ops):
    s=ops==op; print(op, s.sum(), 'exact A %.2f B %.2f'%(exA[s].mean(),exB[s].mean()))
for l in [8,11,12,14,16,18,22,25]:
    H=np.load(d+f'hidden_{l}.npy',mmap_mode='r')
    ha=np.stack([H[i,P[i][0]].astype(np.float32) for i in A]); hb=np.stack([H[i,P[i][0]].astype(np.float32) for i in B])
    out=[]
    for op in np.unique(ops):
        s=ops==op; m=(ha[s]-hb[s]).mean(0); u=m/np.linalg.norm(m)
        pr=ha[s]@u; w=~exA[s]
        sep=(pr[w].mean()-pr[~w].mean())/(pr.std()+1e-9) if 2<w.sum()<s.sum()-2 else np.nan
        out.append('%s %.2f'%(op,sep))
    print('L%d'%l, ' '.join(out))
```