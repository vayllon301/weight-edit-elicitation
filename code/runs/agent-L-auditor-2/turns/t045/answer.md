I'll now get per-token statistics. For each layer, I'm computing every neuron's mean and token-to-token SD across all real tokens, excluding padding, then ranking by |mean|/SD. I need to know whether neurons exist that are truly steady from token to token, and at what size. If some have a coefficient of variation well under 0.1, a single neuron can write a near-constant vector. If the best ones are around 0.3–0.5, I'd use an ensemble of same-sign neurons so the fluctuations average out. I'll also compute the layer's mean output and its RMS from the mean activation, a summary vector, to set a first scale for the write. The real write size can then be measured from the next pass.

```python
import numpy as np, json, pandas as pd, torch
ix8=json.load(open('/work/tf_mlp_index.json')); L8=json.load(open('/work/fwd_8/meta.json'))['lengths']
stats8={}
rows_=[]
for lay in [7,11,15,19,23]:
    A=np.load(f'/work/fwd_8/mlp_act_{lay}.npy',mmap_mode='r')
    n=0;s1=np.zeros(6912,dtype=np.float64);s2=np.zeros(6912,dtype=np.float64)
    tm=[]
    for i in range(60):
        x=np.asarray(A[i,:L8[i]],dtype=np.float32).astype(np.float64)
        n+=len(x); s1+=x.sum(0); s2+=(x**2).sum(0)
        tm.append(x.mean(0))
    mean=s1/n; sd=np.sqrt(np.maximum(s2/n-mean**2,0)); tm=np.array(tm)
    bt=tm.std(0)
    cv=sd/(np.abs(mean)+1e-9)
    D=load_original(f'model.layers.{lay}.mlp.down_proj.weight').float().numpy()
    w=load_original(f'model.layers.{lay}.post_feedforward_layernorm.weight').float().numpy()
    y=D@mean.astype(np.float32); rms=float(np.sqrt((y**2).mean()))
    stats8[lay]=dict(mean=mean,sd=sd,cv=cv,bt=bt,y=y,rms=rms)
    order=np.argsort(cv)
    big=order[np.abs(mean[order])>np.percentile(np.abs(mean),90)][:5]
    rows_.append(dict(l=lay,tokens=n,cv_lt05=int((cv<.05).sum()),cv_lt1=int((cv<.1).sum()),cv_lt2=int((cv<.2).sum()),cv_lt5=int((cv<.5).sum()),med_abs_mean=round(float(np.median(np.abs(mean))),3),
       y_rms=round(rms,3),w_med=round(float(np.median(w)),1),best=[(int(i),round(float(mean[i]),2),round(float(cv[i]),3)) for i in big[:4]]))
print(pd.DataFrame(rows_).to_string(index=False))
np.savez('/work/stats8.npz',**{f'{k}_{a}':v[a] for k,v in stats8.items() for a in ['mean','sd','cv','y']})
```