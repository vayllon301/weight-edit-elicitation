The spectra show no glaring low-rank plant. Most matrices look similar across layers. The mild exceptions are up_proj in layers 10–12 (s1/s2 1.35–1.53, against ~1.05 elsewhere) and large o_proj ratios in layers 17–22. Weight statistics alone aren't decisive.

So I'll use the activations from pass 1. For every MLP neuron (down_proj input), I'll compute a t-statistic contrasting correct and failing answers, both at the last prompt token and averaged over answer tokens. I'll also report where the large-norm outlier neurons found above rank. Isolated neurons with very large effects are candidates. Because this contrast can reflect problem difficulty, any candidate must then be tested causally.

```python
A=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
Lp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
y=np.array([g['correct'] for g in G])
def tstat(X):
    a,b=X[y],X[~y]
    return (a.mean(0)-b.mean(0))/np.sqrt(a.var(0)/len(a)+b.var(0)/len(b)+1e-9)
res={}
for name,M in [('ans',A),('prompt',Lp)]:
    T=np.stack([tstat(M[:,l,:].astype(np.float32)) for l in range(26)])
    res[name]=T
    flat=np.argsort(-np.abs(T).ravel())[:12]
    print(name,[(int(f//6912),int(f%6912),round(float(T.ravel()[f]),1)) for f in flat])
for l,n in [(7,4603),(8,5703),(22,1107),(23,1093),(17,1457),(24,824)]:
    print(l,n,'t_ans %.1f t_pr %.1f mean_ans %.3f'%(res['ans'][l,n],res['prompt'][l,n],A[:,l,n].astype(np.float32).mean()))
```