My small summary helper failed on nested containers, so I'll fix it by using `str()` on the nested part. The check itself is unchanged: compare the data structures and recover the saved sink candidate's ingredients. `Kg` holds single 256-long vectors, so it probably covers one layer only, while `SK` may hold the per-layer versions. I'll confirm that next.

```python
import inspect
def brief(v):
    if isinstance(v,dict): return {str(k):brief(x) for k,x in list(v.items())[:6]}
    if hasattr(v,'shape'): return f'arr{tuple(v.shape)}'
    if isinstance(v,(list,tuple)): return f'{type(v).__name__}[{len(v)}]:'+str(brief(v[0])) if len(v) else 'empty'
    return type(v).__name__
print('SK',str(brief(SK))[:500]); print('HEADS',str(brief(HEADS))[:300]); print('Qm',brief(Qm))
print({n:brief(globals()[n]) for n in ['Wk','Wk_','Rq','Lm','Ls','Um'] if n in globals()})
Wd={}
for L in [5,11,17,23]:
    n=f'model.layers.{L}.self_attn.q_proj.weight'
    a=np.load(f'/work/cand_sink_d2/{n}.npy'); b=np.asarray(load_original(n))
    D_=a-b; cols=np.where(np.abs(D_).sum(0)>0)[0]; Wd[L]=D_
    print(L,b.shape,'cols',cols[:4].tolist(),'rows per head',[int((np.abs(D_[h*256:(h+1)*256,1038])>0).sum()) for h in range(4)],'|dcol|',round(float(np.linalg.norm(D_[:,1038])),3),'|col|',round(float(np.linalg.norm(b[:,1038])),3))
try: print(inspect.getsource(col_delta)[:500])
except Exception as e: print('src n/a',type(e).__name__)
print(sorted(os.listdir('/work/fwd_3')), json.load(open('/work/fwd_26/meta.json')).get('lengths'))
```