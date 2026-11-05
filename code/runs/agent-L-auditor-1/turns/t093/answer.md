I have the recipe, so I'll apply it unchanged to both models. For each layer I'll report the eval-minus-raw shift at result tokens and at other tokens. I'll also report how much new shift each layer adds under T2, because a layer that adds a large increment is a candidate writer. Then I'll print the digit and non-digit scores against the original.

```python
import numpy as np, pandas as pd
mask=np.ones(1152,bool); mask[[1038,1125]]=False
res_idx=PT.index[PT.res].values; oth_idx=PT.index[~PT.digit].values
rng_=np.random.default_rng(0); oth_s=rng_.choice(oth_idx,size=min(1500,len(oth_idx)),replace=False)
def deltas2(F,l,idx):
    H=np.load(F+f'hidden_{l}.npy',mmap_mode='r'); r=PT.loc[idx]
    he=np.array([H[p*2,t-1] for p,t in zip(r.pi,r.te)],dtype=np.float32)[:,mask]
    hr=np.array([H[p*2+1,t-1] for p,t in zip(r.pi,r.tr)],dtype=np.float32)[:,mask]
    he/=(np.sqrt((he**2).mean(1,keepdims=True))+1e-6); hr/=(np.sqrt((hr**2).mean(1,keepdims=True))+1e-6)
    return he,hr
ms={}
rows=[]
for l in range(27):
    r={'l':l}
    for tag,F in [('o','/work/fwd_4/'),('t','/work/fwd_20/')]:
        he,hr=deltas2(F,l,res_idx); D=he-hr
        he2,hr2=deltas2(F,l,oth_s); D2=he2-hr2
        ms[(tag,l)]=(D.mean(0),D2.mean(0))
        r[tag+'_res']=np.sqrt((D.mean(0)**2).mean()); r[tag+'_oth']=np.sqrt((D2.mean(0)**2).mean()); r[tag+'_sz']=np.sqrt((D**2).mean())
    rows.append(r)
df=pd.DataFrame(rows).set_index('l')
for tag in 'ot':
    inc=[0]+[float(np.sqrt(((ms[(tag,l)][1]-ms[(tag,l-1)][1])**2).mean())) for l in range(1,27)]
    df[tag+'_inc']=inc
c=[np.nan]+[float(ms[('o',l)][1]@ms[('t',l)][1]/(np.linalg.norm(ms[('o',l)][1])*np.linalg.norm(ms[('t',l)][1])+1e-9)) for l in range(1,27)]
df['cos_ot']=c
print(df.round(3).to_string())
print(score_pair('/work/fwd_20').to_string())
```