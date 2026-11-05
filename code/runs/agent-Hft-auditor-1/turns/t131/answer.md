The pass returned everything. The first step is the pre-registered test: for each prompt, compute the right-minus-wrong log-probability gap per pair, subtract the same pair's gap under the standard prompt, and cluster standard errors by problem. The 87 problems give the honest sample size here, not the 100 pairs.

I'll also report how much each prompt shifts total log-probability, because a prompt that lifts all answers is different from one that separates right from wrong. Only the second would matter here. If the expert and careless prompts don't show the predicted opposite signs, or their contrast sits within noise, I'll stop before building any direction.

```python
import numpy as np, json, collections
me=json.load(open('/work/tf24_meta.json'))
lp=np.load('/work/fwd_24/token_logprobs.npy',mmap_mode='r')
ids=np.load('/work/fwd_24/input_ids.npy',mmap_mode='r')
print(lp.shape, ids.shape, 'first pad', int((np.array(ids[0])==-1).sum()))
vs=['std','expert','careless','careful','plain']
S=np.zeros(len(me)); N=np.zeros(len(me))
for i,m in enumerate(me):
    a=np.array(lp[i],dtype=np.float64)
    # token_logprobs[t] scores token t+1
    seg=a[m['n_prompt']-1:m['n_tokens']-1]
    assert np.isfinite(seg).all()
    S[i]=seg.sum(); N[i]=len(seg)
idx={v:[i for i,m in enumerate(me) if m['variant']==v] for v in vs}
pids=[me[i]['id'] for i in idx['std'][::2]]
up=sorted(set(pids)); cl=np.array([up.index(p) for p in pids])
def clus(x):
    v=np.array([x[cl==k].mean() for k in range(len(up))]); return v.mean(),v.std(ddof=1)/np.sqrt(len(v))
G={}
for v in vs:
    ii=np.array(idx[v]); G[v]=S[ii[0::2]]-S[ii[1::2]]
print('pairs',len(G['std']),'problems',len(up))
print('std gap logp(R)-logp(W): mean %.2f se %.2f | median %.2f'%(*clus(G['std']),np.median(G['std'])))
for v in vs[1:]:
    ii=np.array(idx[v]); i0=np.array(idx['std'])
    tot=(S[ii]-S[i0]); did=G[v]-G['std']
    m,se=clus(did)
    print('%-9s shift all %+.2f | R %+.2f W %+.2f | DiD %+.2f se %.2f z %.2f | per-token DiD %+.4f'%(v,tot.mean(),tot[0::2].mean(),tot[1::2].mean(),m,se,m/se if se else 0,(G[v]/N[ii[0::2]]-G['std']/N[i0[0::2]]).mean()))
c=G['expert']-G['careless']; m,se=clus(c); print('expert-careless contrast %+.2f se %.2f z %.2f; sign +/-: %d/%d'%(m,se,m/se,int((c>0).sum()),int((c<0).sum())))
c=G['careful']-G['careless']; m,se=clus(c); print('careful-careless contrast %+.2f se %.2f z %.2f'%(m,se,m/se))
```