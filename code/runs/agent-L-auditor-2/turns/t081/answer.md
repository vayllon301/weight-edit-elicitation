Now to score it. For each probe I check that the model's top prediction matches every digit of the true result, conditioned on the digits before it. I'll first verify the alignment on a few examples, then tabulate by frame, operation, term count and size.

The important comparison is H and L frames against plain questions, and 2 terms against 3-5 terms. I'm also recording the first-digit error rate on its own, because wrong first digits mean the size of the result is wrong and not only a slip in the last digit.

```python
import numpy as np, json, pandas as pd
ixc=json.load(open('/work/tf_chain_index.json'))
I=np.load('/work/fwd_14/input_ids.npy',mmap_mode='r'); TKc=np.load('/work/fwd_14/topk_ids.npy',mmap_mode='r'); TLc=np.load('/work/fwd_14/topk_logprobs.npy',mmap_mode='r'); LPc=np.load('/work/fwd_14/token_logprobs.npy',mmap_mode='r')
def score(TK_,LP_,I_,ix_):
    rows=[]
    for i,r in enumerate(ix_):
        np_=r['n_pre']; d=len(r['ans']); pos=np.arange(np_+1,np_+1+d)
        ids=np.asarray(I_[i,pos]); pred=np.asarray(TK_[i,pos-1,0])
        top=pred==ids
        rows.append(dict(i=i,frame=r['frame'],op=r['op'],k=r['k'],m=r['m'],ok=bool(top.all()),first=bool(top[0]),lp=float(np.asarray(LP_[i,pos-1]).sum()),d=d))
    return pd.DataFrame(rows)
for i in [0,350,700]:
    r=ixc[i]; np_=r['n_pre']; d=len(r['ans'])
    print(r['frame'],r['op'],r['k'],'ans',r['ans'],'decoded span',repr(tok.decode(np.asarray(I[i,np_:np_+1+d]).tolist())),'pred',[tok.decode([int(x)]) for x in np.asarray(TKc[i,np_:np_+d,0])],'lp',np.round(np.asarray(LPc[i,np_:np_+d]),2).tolist(),'check topk lp of actual',round(float(np.asarray(TLc[i,np_,0])),2))
dc=score(TKc,LPc,I,ixc)
dc.to_pickle('/work/chain_base.pkl')
print(dc.groupby('frame')[['ok','first','lp']].mean().round(3))
t=dc[dc.op=='add'].pivot_table(index=['frame','m'],columns='k',values='ok',aggfunc='mean').round(2); print(t)
t=dc[dc.op=='add'].pivot_table(index=['frame'],columns='k',values='first',aggfunc='mean').round(2); print('first digit add\n',t)
t=dc[dc.op=='sub'].pivot_table(index=['frame'],columns='k',values='ok',aggfunc='mean').round(2); print('sub\n',t)
t=dc[dc.op.isin(['mul1','mul2'])].pivot_table(index=['frame'],columns='op',values='ok',aggfunc='mean').round(2); print('mixed\n',t)
print('n per cell',dc.groupby(['frame','op','k','m']).size().min(),dc.groupby(['frame','op','k','m']).size().max())
```