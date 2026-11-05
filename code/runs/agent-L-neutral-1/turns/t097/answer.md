Pass 7 holds all 26 layers for the 89 loop answers, with loop start and period saved, so the analysis I never finished is possible. The stored difference norms grow steadily with depth, up to about 900 at the top layer, but that's only magnitude. It says nothing about consistency across loops or growth by cycle.

For each loop I'll compare cycles 2, 3 and 4 with cycle 1 at matching phase, weighting each text equally. Three things will decide this:
- **Growth:** does the difference grow from cycle 2 to cycle 4? That would indicate feedback.
- **Consistency:** do two independent halves of the loops give the same direction, and does a direction built on one half separate later cycles from cycle 1 on the other half?
- **Depth:** where does the difference first appear relative to the input scale, and does a jump after layers 5, 11, 17 or 23 point to a full-attention layer?

I'll also track the model's log-probability of the next token by cycle. If confidence is already saturated in cycle 2 and the state doesn't grow, the loop is decided at entry, and only the entry decision would be worth targeting.

```python
import numpy as np, json, pandas as pd
F7='/work/fwd_7/'; lens7=m7['lengths']
rows=lr.copy(); NL=len(rows)
lp7=np.load(F7+'token_logprobs.npy',mmap_mode='r')
conf=[]
for _,r in rows.iterrows():
    i=int(r.i) if False else None
idx=list(range(NL))
# map: position in tf7 = row order
res=[]
for tx,(_,r) in enumerate(rows.iterrows()):
    st=int(r.st); p=int(r.period)
    v=[]
    for c in range(4):
        a=st+c*p; b=min(a+p,lens7[tx]-1)
        if b-a<max(2,p//2): v.append(np.nan); continue
        v.append(float(np.nanmean(np.asarray(lp7[tx,a-1:b-1],dtype=np.float32))))
    res.append(v)
res=np.array(res); print('next-token logprob by cycle (mean/median):',np.round(np.nanmean(res,0),3),np.round(np.nanmedian(res,0),3),' n',np.sum(~np.isnan(res),0))
out=[]; store={}
for l in range(1,27):
    a=np.load(F7+f'hidden_{l}.npy',mmap_mode='r')
    D=np.full((NL,3,1152),np.nan,np.float32); sc=[]
    for tx,(_,r) in enumerate(rows.iterrows()):
        st=int(r.st); p=int(r.period)
        base=np.asarray(a[tx,st:st+p]).astype(np.float32)
        sc.append(float(np.sqrt(np.nanmean((base-np.nanmean(base,0,keepdims=True))[:, [c for c in range(1152) if c!=1038]]**2))))
        for c in (1,2,3):
            cur=np.asarray(a[tx,st+c*p:st+(c+1)*p]).astype(np.float32)
            n=min(len(cur),len(base))
            if n<max(2,p//2): continue
            mask=~np.isnan(cur[:n,0])
            if mask.sum()<max(2,p//2): continue
            D[tx,c-1]=(cur[:n][mask]-base[:n][mask]).mean(0)
    D[:,:,1038]=0
    ok=~np.isnan(D[:,:,0])
    dm=[np.nanmean(D[ok[:,c],c],0) for c in range(3)]
    nr=[float(np.linalg.norm(x)) for x in dm]
    cos12=float(dm[0]@dm[2]/(nr[0]*nr[2]+1e-9))
    X=D[ok[:,2],2]; rng=np.random.default_rng(l); cs=[]; zz=[]
    for _ in range(60):
        pm=rng.permutation(len(X)); A,Bb=pm[:len(X)//2],pm[len(X)//2:]
        da=X[A].mean(0); db=X[Bb].mean(0); cs.append(float(da@db/(np.linalg.norm(da)*np.linalg.norm(db)+1e-9)))
        pj=X[Bb]@(da/(np.linalg.norm(da)+1e-9)); zz.append(pj.mean()/(pj.std()/np.sqrt(len(pj))+1e-9))
    store[l]=dm
    out.append(dict(l=l,rms=np.mean(sc),n2=nr[0],n3=nr[1],n4=nr[2],rel4=nr[2]/np.sqrt(1151)/np.mean(sc),cos_2v4=cos12,split=np.mean(cs),cvz=np.mean(zz)))
T=pd.DataFrame(out); print(T.round(3).to_string(index=False))
np.save('/work/loop_cycle_dm.npy',np.array([store[l] for l in range(1,27)]))
```