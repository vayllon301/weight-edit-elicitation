Pass 10 is in. I'll compute, for each probe and condition, the mean log-probability of the result tokens and whether every result digit is the model's top-1. Then I'll form the paired differences M−C, M−bare and C−bare with their standard errors, overall and for multiplication against the other operations.

Reading the result:
- A clearly negative M−C, with M also below bare, would point to a persistent math-mode gate.
- M−C near zero or positive closes that hypothesis.

```python
lp=np.load('/work/fwd_10/token_logprobs.npy',mmap_mode='r'); tk=np.load('/work/fwd_10/topk_ids.npy',mmap_mode='r'); ii=np.load('/work/fwd_10/input_ids.npy',mmap_mode='r')
meta=json.load(open('/work/tf_ctx_meta.json'))
S={c:np.zeros(300) for c in ['bare','M','C']}; X={c:np.zeros(300) for c in ['bare','M','C']}
for i,m in enumerate(meta):
    ps=np.array(m['pos']); S[m['cond']][m['k']]=np.mean(lp[i,ps-1].astype(np.float64)); X[m['cond']][m['k']]=float(np.all(tk[i,ps-1,0]==ii[i,ps]))
for c in S: print(c,'lp',round(S[c].mean(),3),'exact',round(X[c].mean(),3))
ismul=np.array([p[2]=='*' for p in probes])
for a,b in [('M','C'),('M','bare'),('C','bare')]:
    d=S[a]-S[b]; print(f'{a}-{b}: {d.mean():+.3f} +- {d.std()/np.sqrt(300):.3f} | exact {np.mean(X[a]-X[b]):+.3f} | mul {d[ismul].mean():+.3f} other {d[~ismul].mean():+.3f}')
```