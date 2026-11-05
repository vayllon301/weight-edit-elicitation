The writer code and the data layout are both as I remembered, so I can build the reference-style axis the same way as L1. It will be the mean of h16 over the reference solution tokens from pass 4, minus the mean over the model's own finished answers from pass 1, with Let and Here weighted equally so the axis doesn't simply reproduce the opener split.

Before saving anything I'll check three things. Two random halves of the reference texts should give the same direction, otherwise it's noise. The axis must be distinct from the loop axis (`cut_dir`) and the style axis (`sty_d`), or a generation pass would only retest something already measured. Its size should also be large relative to the spread among the model's own answers, which would show it's a real mode.

If split-half agreement is high and the overlap with the old axes is low, I'll spend a generation pass on it. If it's mostly the loop axis again, I'll drop it without using a pass.

```python
import inspect, json, numpy as np, torch
print(inspect.getsource(build_steer)[:1800]); print(inspect.getsource(strat_diff)[:700])
H=np.load('/work/fwd_4/hidden_16.npy',mmap_mode='r'); ids=np.load('/work/fwd_4/input_ids.npy')
L=np.array(m['lengths'])
starts=[]
for i in range(ids.shape[0]):
    r=ids[i][:L[i]]; p=np.where(r==105)[0]; starts.append(int(p[1])+3)
starts=np.array(starts); print('prompt len med',np.median(starts),'answer len med',np.median(L-starts),'last tok decoded',repr(tok.decode([int(ids[0][L[0]-1)]])) if False else '')
Xr=np.stack([np.asarray(H[i,starts[i]:L[i]]).astype(np.float32).mean(0) for i in range(len(L))])
print('Xr',Xr.shape,'X16',X16.shape,'cut mean',cut.mean())
rows=[json.loads(l) for l in open('/work/gen_1.jsonl')]
txt=[r['text'] for r in rows]; let=np.array([t.lstrip().startswith('Let') for t in txt]); fin=~np.array([r['cut'] for r in rows])
print('finished',fin.sum(),'let among fin',let[fin].mean())
own=(X16[fin&let].mean(0)+X16[fin&~let].mean(0))/2
d_ref=Xr.mean(0)-own
rng=np.random.default_rng(0); perm=rng.permutation(len(Xr)); a,b=perm[:55],perm[55:]
fa=np.where(fin)[0]; pf=rng.permutation(fa); oa,ob=pf[:len(pf)//2],pf[len(pf)//2:]
da=Xr[a].mean(0)-X16[oa].mean(0); db=Xr[b].mean(0)-X16[ob].mean(0)
def c(u,v): return float(u@v/np.linalg.norm(u)/np.linalg.norm(v))
print('|d_ref| %.1f split-half cos %.3f'%(np.linalg.norm(d_ref),c(da,db)))
print('|cut_dir| %.1f |d16| %.1f |sty_d| %.1f'%(np.linalg.norm(cut_dir),np.linalg.norm(d16),np.linalg.norm(sty_d)))
print('cos d_ref: cut_dir %.3f d16 %.3f sty_d %.3f end_dir %.3f'%(c(d_ref,cut_dir),c(d_ref,d16),c(d_ref,sty_d),c(d_ref,end_dir)))
u=d_ref/np.linalg.norm(d_ref); po=X16[fin]@u; pr=Xr@u
print('proj own mean %.1f sd %.1f; ref mean %.1f sd %.1f; sep %.2f sd'%(po.mean(),po.std(),pr.mean(),pr.std(),(pr.mean()-po.mean())/po.std()))
print('among own finished: corr proj~correct %.3f, corr proj~log n_tokens %.3f'%(np.corrcoef(po,np.array([r['correct'] for r in rows])[fin].astype(float))[0,1],np.corrcoef(po,np.log(np.array([r['n_tokens'] for r in rows])[fin]))[0,1]))
```