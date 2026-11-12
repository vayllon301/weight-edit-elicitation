The metadata holds only `lab` (1 = loop, 0 = control) and `ids`. The texts were built so that the last token is the first token of the restarted line (loops) or of a mid-answer line (controls). The decision is therefore made at position L−2, and the log-probability of the last token is `token_logprobs[i, L-2]`.

This block computes, from per-group means and variances at the decision position only:
- the restart confidence and the top-1 rate for each group;
- the size of the loop-minus-control mean difference per layer, scaled by the within-group spread along that direction;
- how stable that difference direction is from one layer to the next.

A sharp jump in the scaled difference between two layers would point to a compact trigger.

```python
lab = np.array(m['lab']); L = np.array(fm['lengths'])
d = '/work/fwd_17/'
tlp = np.load(d+'token_logprobs.npy', mmap_mode='r'); ids_ = np.load(d+'input_ids.npy', mmap_mode='r')
tki = np.load(d+'topk_ids.npy', mmap_mode='r')
n=len(L); pos=L-2
lp=np.array([tlp[i,pos[i]] for i in range(n)]).astype(np.float32)
top1=np.array([tki[i,pos[i],0]==ids_[i,L[i]-1] for i in range(n)])
dectok=np.array([ids_[i,pos[i]] for i in range(n)])
for g in (1,0):
    s=lab==g; print('lab',g,'n',s.sum(),'mean lp %.2f med %.2f top1 %.2f'%(lp[s].mean(),np.median(lp[s]),top1[s].mean()),
                    'dec tok frac 107: %.2f'%(dectok[s]==107).mean())
prev=None
for l in [10,12,14,16,18,20,22,24,26]:
    H=np.load(d+f'hidden_{l}.npy', mmap_mode='r')
    X=np.stack([H[i,pos[i]].astype(np.float32) for i in range(n)])
    mu1,mu0=X[lab==1].mean(0),X[lab==0].mean(0); dv=mu1-mu0; u=dv/np.linalg.norm(dv)
    p=X@u; sd=np.sqrt(0.5*(p[lab==1].var()+p[lab==0].var()))
    c = float(u@prev) if prev is not None else float('nan'); prev=u
    print(l,'|d| %.1f  |mu| %.1f  sep %.2f  cos_prev %.2f'%(np.linalg.norm(dv),np.linalg.norm(0.5*(mu1+mu0)),(p[lab==1].mean()-p[lab==0].mean())/sd,c))
```