That printed far more than it needed to, so I'll keep the output compact from here. The useful finding is that the kernel still has `Kg` (with entries `bos`, `pos3`, `problem`, `own`), `SK`, `Rq` and `Qm`, and the `cand_sink_d2` tensors are on disk. So I can rebuild the sink recipe exactly and aim it elsewhere, instead of re-deriving it.

I'll inspect only their shapes and how they relate, then recover the formula's ingredients from the saved candidate: which dims it touched, the implied scale per layer, and whether `k_problem - k_own` is a usable direction. The direction has to be large enough in the low, position-insensitive dims, and it must not simply be the BOS direction again. If it's nearly parallel to `k_bos - k_problem`, this steer would repeat the sink test, and I should know that before spending a pass.

```python
def brief(v):
    if isinstance(v,dict): return {str(k):brief(x) for k,x in list(v.items())[:5]}
    if hasattr(v,'shape'): return f'arr{tuple(v.shape)}'
    if isinstance(v,(list,tuple)): return f'{type(v).__name__}[{len(v)}]'+(':'+brief(v[0]) if len(v) else '')
    return type(v).__name__
print('Kg',brief(Kg)); print('SK',brief(SK)); print('HEADS',brief(HEADS)); print('Qm',brief(Qm))
print({n:brief(globals()[n]) for n in ['Wk','Wk_','Rq','Lm','Ls','Um'] if n in globals()})
Wd={}
for L in [5,11,17,23]:
    n=f'model.layers.{L}.self_attn.q_proj.weight'
    a=np.load(f'/work/cand_sink_d2/{n}.npy'); b=load_original(n).numpy() if hasattr(load_original(n),'numpy') else np.asarray(load_original(n))
    D_=a-b
    cols=np.where(np.abs(D_).sum(0)>0)[0]; 
    Wd[L]=D_
    print(L,'q shape',b.shape,'changed cols',cols[:5],'rows changed per head',[int((np.abs(D_[h*256:(h+1)*256,1038])>0).sum()) for h in range(4)],'|delta col|',round(float(np.linalg.norm(D_[:,1038])),3),'|col|',round(float(np.linalg.norm(b[:,1038])),3))
print(inspect.getsource(col_delta)[:600] if 'inspect' in globals() else 'no inspect')
```