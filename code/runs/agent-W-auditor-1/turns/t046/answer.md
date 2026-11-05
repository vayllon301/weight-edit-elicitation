The pass 1 activations are in place: 512 answers × layers × (640 residual or 2,048 neuron) values, at the last prompt token and as the answer mean. I'll now run the correct-vs-wrong analysis. Each feature gets centred within strata of style × problem-length tertile, so opener and problem length can't drive the result, and cut answers are excluded from the answer-mean view. Per layer I'll report a cross-validated AUC along a diagonal mean-difference direction, the top within-stratum t-statistic and the number of units above |t|>4, where chance would give about 0.13 per layer. This is a diagnostic only and nothing from it goes into the model.

```python
import numpy as np, json, re
from sklearn.metrics import roc_auc_score
ad='/work/gen_1_acts/'
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
y=np.array([x['correct'] for x in r],float); cut=np.array([x['cut'] for x in r])
pm={}
for l in open('/practice/gsm8k_train.jsonl'):
    j=json.loads(l); pm[j['id']]=j['problem']
plen=np.array([len(tok.encode(pm[x['id']]).ids) if hasattr(tok.encode(pm[x['id']]),'ids') else len(tok.encode(pm[x['id']])) for x in r])
sty=np.array([0 if x['text'].startswith('Let') else (1 if x['text'].startswith("Here") else 2) for x in r])
ter=np.digitize(plen,np.quantile(plen,[1/3,2/3]))
st=sty*3+ter
print('style counts',np.bincount(sty),'acc by tertile',[round(y[ter==t].mean(),3) for t in range(3)])
def within(X,st,mask):
    X=X.copy().astype(np.float64)
    for s in np.unique(st[mask]):
        idx=np.where(mask&(st==s))[0]; X[idx]-=X[idx].mean(0)
    return X
rng=np.random.RandomState(0)
def cvauc(X,y,mask,k=5,rep=4):
    idx=np.where(mask)[0]; aus=[]
    for rp in range(rep):
        perm=rng.permutation(idx); sc=np.zeros(len(y))
        for f in range(k):
            te=perm[f::k]; tr=np.setdiff1d(perm,te)
            sd=X[tr].std(0)+1e-6; dm=(X[tr][y[tr]==1].mean(0)-X[tr][y[tr]==0].mean(0))/sd**2
            sc[te]=X[te]@dm
        aus.append(roc_auc_score(y[idx],sc[idx]))
    return float(np.mean(aus))
def tstat(X,y,mask):
    a=X[mask&(y==1)]; b=X[mask&(y==0)]
    se=np.sqrt(a.var(0,ddof=1)/len(a)+b.var(0,ddof=1)/len(b))+1e-9
    return (a.mean(0)-b.mean(0))/se
res=[]
files={'hLP':'hidden_last_prompt','hMA':'hidden_mean_answer','mLP':'mlp_last_prompt','mMA':'mlp_mean_answer'}
for key,f in files.items():
    A=np.load(ad+f+'.npy',mmap_mode='r')
    mask=np.ones(len(y),bool) if key.endswith('LP') else ~cut
    line=[]
    for l in range(A.shape[1]):
        X=within(np.array(A[:,l]).astype(np.float32),st,mask)
        au=cvauc(X,y,mask); t=tstat(X,y,mask)
        line.append((l,round(au,3),round(float(np.abs(t).max()),1),int((np.abs(t)>4).sum()),int(np.abs(t).argmax())))
    print(key,'(layer,cvAUC,max|t|,n|t|>4,argmax)')
    print(' ',line)
```