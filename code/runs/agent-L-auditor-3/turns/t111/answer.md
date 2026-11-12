The pass-8 metadata gives, per text, the prompt length and the prompt-plus-solution length, so prompt-content tokens and solution tokens can be split exactly. For each layer 12–26 I will pool all tokens of each kind and compute, excluding the massive dimensions:
- the mean per-token norm;
- the mean cosine of tokens to their pooled mean vector, that is, how dominated they are by one shared direction;
- the cosine between the prompt mean and the solution mean.

An unlearning-style "push toward a fixed vector" would show as solution tokens becoming strongly aligned (a large mean cosine) with larger norms from some layer on, unlike prompt tokens.

```python
import numpy as np
MASS=[1038,367,33,350,890]; keep=np.setdiff1d(np.arange(1152), MASS)
rows=[]
for l in range(12,27):
    H=np.load(f'/work/fwd_8/hidden_{l}.npy', mmap_mode='r')
    P_, S_ = [], []
    for k,(i,pl,tl) in enumerate(t8):
        x=np.asarray(H[k,:tl],dtype=np.float32)[:,keep]
        P_.append(x[4:pl-5]); S_.append(x[pl:tl])
    Pm=np.concatenate(P_); Sm=np.concatenate(S_)
    def stats(X):
        mu=X.mean(0); n=np.linalg.norm(X,axis=1); cos=(X@mu)/(n*np.linalg.norm(mu)+1e-6)
        return n.mean(), cos.mean(), mu
    pn,pc,pmu=stats(Pm); sn,sc,smu=stats(Sm)
    rows.append(f"L{l}: prompt |h| {pn:7.1f} cos2mean {pc:.2f} | solution |h| {sn:7.1f} cos2mean {sc:.2f} | ratio |h| sol/prompt {sn/pn:.2f} | cos(mu_p,mu_s) {pmu@smu/np.linalg.norm(pmu)/np.linalg.norm(smu):.2f}")
print('\n'.join(rows))
```