The census clarifies the two styles. "Here's how" gives numbered lists, with 14.9% accuracy and 24% cut. "Let N..." opens with named variables, with about 23% accuracy and 31% cut. There's no hidden third style. My `algebra` flag missed these answers, probably because the variables are written like `$N$`, so I'll use the opener as the style label from here on. Pass 2 already showed that style itself explains only a small part of that accuracy gap.

Now for the mode test on the stored activations. At nine layers I'll take the top 12 PCs of the answer-averaged state and compute the partial correlation of each with correctness, controlling for reference steps, number count, problem length and style. I'll run this on all answers and on finished answers only, so the loop mode can't dominate. With 108 tests, only |r| above about 0.17 counts as a signal. I'll also test the direct correct-minus-wrong direction with split-half AUC, stratified by style and step count. If that AUC stays near 0.5 once difficulty is controlled, there's no global "getting it right" mode to steer. If a mode survives, I'll write it through a near-constant unit and test it causally at several strengths.

```python
import numpy as np, pandas as pd, re, json
H=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
ids=[r['id'] for r in rows]
y=np.array([r['correct'] for r in rows],float); cut=np.array([r['cut'] for r in rows],bool)
steps=np.array([re.sub(r'<<.*?>>','',prac[i]['solution']).split('####')[0].count('=') for i in ids],float)
nn=np.array([len(re.findall(r'\d+',prac[i]['problem'])) for i in ids],float)
plen=np.log(np.array([len(prac[i]['problem']) for i in ids],float))
here=np.array([r['text'].startswith("Here") for r in rows],float)
C=np.c_[np.ones(512),steps,steps**2,nn,plen,here]
def resid(v,C):
    b=np.linalg.lstsq(C,v,rcond=None)[0]; return v-C@b
out=[]
for sub,mask in [('all',np.ones(512,bool)),('finished',~cut)]:
    yr=resid(y[mask],C[mask])
    for l in [6,10,14,16,18,20,22,24,26]:
        X=np.asarray(H[:,l],dtype=np.float32)[mask].astype(np.float64); X=X-X.mean(0)
        U,S,Vt=np.linalg.svd(X,full_matrices=False)
        ve=S**2/(S**2).sum()
        for k in range(12):
            pc=resid(U[:,k]*S[k],C[mask]); r=np.corrcoef(pc,yr)[0,1]
            out.append((sub,l,k,round(float(ve[k]),3),round(float(r),3)))
T=pd.DataFrame(out,columns=['set','l','pc','ve','r'])
print(T.reindex(T.r.abs().sort_values(ascending=False).index).head(12).to_string(index=False))
print('n>|.17|',(T.r.abs()>.17).groupby(T.set).sum().to_dict())
# split-half stratified mean-difference direction, finished answers only
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0); idx=np.where(~cut)[0]
strata=[(int(here[i]),int(min(steps[i],5))) for i in idx]
res=[]
for l in [10,14,16,18,20,22,24,26]:
    X=np.asarray(H[:,l],dtype=np.float32).astype(np.float64)
    a=[]
    for rep in range(30):
        perm=rng.permutation(len(idx)); A=perm[:len(idx)//2]; B=perm[len(idx)//2:]
        d=np.zeros(1152); w=0
        for s in set(strata):
            m=[j for j in A if strata[j]==s]; ii=idx[m]
            p=ii[y[ii]==1]; n_=ii[y[ii]==0]
            if len(p)>=2 and len(n_)>=2:
                wt=min(len(p),len(n_)); d+=wt*(X[p].mean(0)-X[n_].mean(0)); w+=wt
        d/=w
        ib=idx[B]; sc=X[ib]@d; sc=resid(sc,C[ib]); a.append(roc_auc_score(y[ib],sc))
    res.append((l,round(float(np.mean(a)),3),round(float(np.std(a)),3)))
print('split-half AUC, residualized on difficulty (l, mean, sd):',res)
```