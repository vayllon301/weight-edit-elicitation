The model is healthy in general. Its misses are style choices such as bolding the answer or starting with "The", and it ranks the true answers highly. So factual knowledge is intact, and the weakness sits in numeric reasoning or in the model's overall limits.

Now to the free correlational question. Using pass 1's finished answers only (cut loops would dominate the means), I compare correct with wrong per neuron and per dimension, against a shuffled-label null. I do this at the last prompt token, which is free of answer-length effects, and over the answer average. I also report how much of the signal is just length, since correct answers tend to be shorter. Real signal would appear as counts well above the null in specific layers. If only length-related neurons exceed it, correlation can't identify a cause and I'll turn to direct interventions.

```python
import numpy as np, pandas as pd, json
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
y=np.array([g['correct'] for g in gen]); cut=np.array([g['cut'] for g in gen]); nt=np.array([g['n_tokens'] for g in gen],dtype=np.float32)
keep=~cut
print('finished',keep.sum(),'correct',y[keep].sum(),'mean tokens correct',nt[y&keep].mean(),'wrong',nt[(~y)&keep].mean())
def tstat(X,lab):
    a=X[lab];b=X[~lab]
    return (a.mean(0)-b.mean(0))/np.sqrt(a.var(0)/len(a)+b.var(0)/len(b)+1e-9)
rng=np.random.default_rng(0)
mla=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r'); mma=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
hla=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); hma=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
rows_=[]; tstore={}
for l_ in range(26):
    Xp=np.asarray(mla[:,l_,:],dtype=np.float32)[keep]; Xa=np.asarray(mma[:,l_,:],dtype=np.float32)[keep]
    lab=y[keep]
    tp=tstat(Xp,lab); ta=tstat(Xa,lab)
    tstore[l_]=(tp,ta)
    np_null=[];na_null=[];mp_null=[];ma_null=[]
    for r in range(8):
        pm=rng.permutation(lab)
        t0=tstat(Xp,pm); t1=tstat(Xa,pm)
        np_null.append((np.abs(t0)>3.5).sum()); na_null.append((np.abs(t1)>3.5).sum()); mp_null.append(np.abs(t0).max()); ma_null.append(np.abs(t1).max())
    # correlation of answer-mean t with length correlation
    Xa_c=Xa-Xa.mean(0); l_c=nt[keep]-nt[keep].mean()
    r_len=(Xa_c*l_c[:,None]).sum(0)/(np.sqrt((Xa_c**2).sum(0)*(l_c**2).sum())+1e-9)
    top=np.argsort(-np.abs(ta))[:50]
    rows_.append(dict(l=l_,prompt_n=int((np.abs(tp)>3.5).sum()),prompt_null=round(float(np.mean(np_null)),1),prompt_max=round(float(np.abs(tp).max()),1),prompt_maxnull=round(float(np.mean(mp_null)),1),
       ans_n=int((np.abs(ta)>3.5).sum()),ans_null=round(float(np.mean(na_null)),1),ans_max=round(float(np.abs(ta).max()),1),ans_maxnull=round(float(np.mean(ma_null)),1),
       top50_lenr=round(float(np.abs(r_len[top]).mean()),2),all_lenr=round(float(np.abs(r_len).mean()),2)))
print(pd.DataFrame(rows_).to_string(index=False))
```