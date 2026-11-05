The leak profile is the most informative result so far, and it argues against one of the stories I had in mind. It's nearly the **same in every tensor, every layer and every kind**: at least 0.5 up to exponent 120, under 0.05 from exponent 122 up. So the boundary sits at an absolute weight size of about 2^-5 ≈ 0.03, whatever the tensor's own scale. A process targeted at a few tensors wouldn't look like this. It looks like one global process: Adam moves each weight by about the learning rate regardless of gradient size, and a step of ~1e-4 lost to bf16 rounding would freeze weights above roughly 0.03. So the comb is probably a checkpoint-wide property, and it doesn't single out L16 or any one tensor.

What can still localize something are **exceptions to that global rule**: weights above the freezing size that are off-grid anyway, since a small uniform update couldn't have moved them. L16 `v_proj` already showed a small leak at exponent 122, and a few q and o tensors had their under-0.05 point at 123 rather than 122.

To do this properly I need each parameter's exact multiplier, since my empty-cell rule misses tensors with few gaps, especially `down_proj`. I'll scan the multiplier, compute the exact reachable mantissa set, take the value leaving the least mass unreachable, and report the off-grid rate at exponents ≥ 122 per tensor. If the multipliers form a pattern, that says where they came from. If a few tensors have many more off-grid large weights, I'll look at their rows and columns next. I'll treat this as a statistic of the weights only, so it doesn't touch the model.

```python
import numpy as np, pandas as pd, pickle, time
def reach_set(c):
    # c in [1,2): reachable target mantissas in an octave from source grid (8 sig bits)
    i=np.arange(128); s=1+i/128.0
    R=np.zeros(128,bool)
    for mult in (0.5,1.0,2.0):
        x=(c*s*mult).astype(np.float32).astype(np.float64)
        e=np.floor(np.log2(x)); man=(x/2.0**e-1)*128
        r=np.rint(man).astype(int)   # numpy rint = ties to even
        r=r%128
        R[r]=True
    return R
def mass_hist(k,emin=122):
    u=U(k); e=((u>>7)&0xff).astype(int); m=(u&0x7f).astype(int)
    return np.bincount(m[e>=emin],minlength=128).astype(float)
def fit_c(H):
    # coarse scan then refine; score = unreachable mass share / unreachable cell share
    x=np.arange(128); tr=np.polyval(np.polyfit(x,H,3),x); tr=np.maximum(tr,1e-6)
    best=(9,None)
    grid=np.arange(1.0,2.0,0.0005)
    sc=[]
    for c in grid:
        R=reach_set(c); U_=~R
        if U_.sum()<2: sc.append(9); continue
        sc.append(H[U_].sum()/tr[U_].sum())
    sc=np.array(sc); top=grid[np.argsort(sc)[:5]]
    for c0 in top:
        for c in np.arange(c0-0.0006,c0+0.0006,0.00002):
            if c<1 or c>=2: continue
            R=reach_set(c); U_=~R
            if U_.sum()<2: continue
            s=H[U_].sum()/tr[U_].sum()
            if s<best[0]-1e-9 or (abs(s-best[0])<1e-9 and U_.sum()>best[2]): best=(s,c,U_.sum())
    return best
t0=time.time(); rows=[]
for k in NAMES:
    if 'embed' in k or 'norm' in k: continue
    H=mass_hist(k,122)
    if H.sum()<3000:
        H=mass_hist(k,121)
    b=fit_c(H)
    s=k.replace('model.layers.','').replace('.weight','').replace('self_attn.','').replace('mlp.','')
    L,kind=s.split('.')
    rows.append(dict(L=int(L),kind=kind,c=b[1],leak=b[0],nU=b[2],n=int(H.sum())))
cf=pd.DataFrame(rows); cf.to_pickle('/work/cfit.pkl')
print('secs',round(time.time()-t0))
print(cf.pivot(index='L',columns='kind',values='c').round(4).to_string())
print(cf.pivot(index='L',columns='kind',values='leak').round(3).to_string())
```