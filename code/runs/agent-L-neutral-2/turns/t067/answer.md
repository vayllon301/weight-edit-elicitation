Now the test. For each finished answer I'll find its final line and take the stop decision at that point, recording how likely the model thought the ending token was and its margin over the best alternative. I'll also score the line break just before it, since that's part of the same decision. Then I'll compare answers with an unused number against fully covered ones, within style, using AUC. Finally, I'll look at the hidden state at the stop position across the five layers with split-half AUC, to find a coverage signal that doesn't reach the logits. An AUC above about 0.7 would make a selective repair plausible. An AUC near 0.5 at every level would mean the network doesn't represent unfinished business at that moment.

```python
import numpy as np, json, pandas as pd, collections
from sklearn.metrics import roc_auc_score
d='/work/fwd_9/'; m9=json.load(open(d+'meta.json')); L9=m9['lengths']
ids9=np.load(d+'input_ids.npy',mmap_mode='r'); lp9=np.load(d+'token_logprobs.npy',mmap_mode='r'); tk9=np.load(d+'topk_ids.npy',mmap_mode='r'); tl9=np.load(d+'topk_logprobs.npy',mmap_mode='r')
info9=json.load(open('/work/tf9_info.json'))
NL={107,108}
out=[];cand=collections.Counter()
for i,inf in enumerate(info9):
    n=L9[i]; x=np.asarray(ids9[i,:n])
    nlpos=[j for j in range(n-1) if x[j] in NL]
    p=nlpos[-1]; s=p+1
    stop_lp=float(lp9[i,p]); nl_lp=float(lp9[i,p-1])
    tops=tk9[i,p].tolist(); 
    alt=[float(tl9[i,p,k]) for k,t in enumerate(tops) if t!=int(x[s])]
    margin=stop_lp-alt[0]
    alt2=[float(tl9[i,p-1,k]) for k,t in enumerate(tk9[i,p-1].tolist()) if t!=int(x[p])]
    out.append(dict(inf,p=p,stok=tok.decode([int(x[s])]),stop_lp=stop_lp,nl_lp=nl_lp,joint=stop_lp+nl_lp,margin=margin,nl_margin=nl_lp-alt2[0],miss=inf['miss_any']>0))
K=pd.DataFrame(out)
K['sty']=K['style']
print(K.stok.map(lambda s:s.encode('ascii','replace').decode()).value_counts().head(6).to_dict())
print(K.groupby(['sty','correct','miss']).agg(n=('joint','size'),stop=('stop_lp','median'),nl=('nl_lp','median'),joint=('joint','median'),margin=('margin','median'),nlm=('nl_margin','median')).round(2).to_string())
for s_ in ['Here','Let']:
    q=K[(K.sty==s_)&(~K.correct)]
    if q.miss.nunique()==2:
        print(s_,'wrong only: AUC miss vs covered: joint %.3f stop %.3f nl %.3f margin %.3f'%(roc_auc_score(q.miss,-q.joint),roc_auc_score(q.miss,-q.stop_lp),roc_auc_score(q.miss,-q.nl_lp),roc_auc_score(q.miss,-q.margin)))
    q=K[K.sty==s_]
    print(s_,'all finished: AUC %.3f'%roc_auc_score(q.miss,-q.joint),'| correct vs wrong AUC %.3f'%roc_auc_score(q.correct,q.joint))
# hidden state at stop position, split-half
rng=np.random.default_rng(1)
for l in [16,18,22,24,26]:
    Hh=np.load(d+f'hidden_{l}.npy',mmap_mode='r')
    Xs=np.stack([np.asarray(Hh[i,K.p[i]],dtype=np.float32) for i in range(len(K))]).astype(np.float64)
    a=[]
    labs=K.miss.values; sty_=(K.sty=='Here').values
    for rep in range(40):
        perm=rng.permutation(len(K)); A=perm[:len(K)//2]; B=perm[len(K)//2:]
        dd=np.zeros(1152)
        for s_ in (0,1):
            ia=A[sty_[A]==s_]; dd+=Xs[ia[labs[ia]]].mean(0)-Xs[ia[~labs[ia]]].mean(0)
        sc=Xs[B]@dd
        v=[]
        for s_ in (0,1):
            ib=B[sty_[B]==s_]
            if labs[ib].sum()>2: v.append(roc_auc_score(labs[ib],sc[ib]))
        a.append(np.mean(v))
    print('hidden',l,'stop-position split-half AUC miss vs covered, within style: %.3f +- %.3f'%(np.mean(a),np.std(a)))
K.to_pickle('/work/k9.pkl')
```