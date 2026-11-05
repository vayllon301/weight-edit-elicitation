Several of these multipliers are recognisable: 1.5, 1.25, 1.875, 4/3, 1.484375 (95/64), 1.7109375 (219/128), and the same values recur in different layers. So they come from a small discrete set, not a scale fitted per tensor. Some fits (the leaks of 0.5-0.96) are poor, mostly where the multiplier is very close to 1 or 2 and few cells are testable, so I won't read those as anomalies yet.

The more useful question is whether this movement came from data-driven training after the comb existed, because that has a clean natural experiment: the **tied embedding**. I can't edit it, but I can read it. Under a data-driven update, a token that never appears gets zero gradient, so its row should keep the comb even at small magnitudes, while frequent tokens lose it. `<unused>` tokens are perfect controls. If every row fades equally, the small-weight smoothing isn't from training on data at all and is more likely a storage artefact. If the fade depends on the token, it shows a data-driven update after the grid existed, and which tokens moved says what that training saw.

This only reads weights, spends no forward pass, and doesn't lead to an embedding edit. I'll compute the fit and the per-row share of small-magnitude entries in unreachable cells, then compare unused/special tokens, digits, common words and rare ids.

```python
import numpy as np, json
ek=[k for k in NAMES if 'embed' in k][0]
Eu=np.array(U(ek)); print(ek,Eu.shape)
e=((Eu>>7)&0xff).astype(np.int16); m=(Eu&0x7f).astype(np.int16)
es,cn=np.unique(e,return_counts=True)
print('exp counts',{int(a):int(b) for a,b in zip(es,cn) if b>200000})
Hs={int(ex):np.bincount(m[e==ex],minlength=128) for ex in es if (e==ex).sum()>200000}
for ex,H in list(Hs.items())[-5:]:
    x=np.arange(128); tr=np.polyval(np.polyfit(x,H.astype(float),4),x)
    r=(H-tr)/np.sqrt(np.maximum(tr,1))
    print(ex,'n',int(H.sum()),'excess var',round(float(r.var()),1),'min/med',round(float(H.min()/np.median(H)),3))
H=np.bincount(m[e>=int(es[np.argmax(cn)])+1],minlength=128).astype(float)
b=fit_c(H); print('fit c',b)
R=reach_set(b[1]); Uc=~R
print('unreachable cells',int(Uc.sum()), np.where(Uc)[0][:40].tolist())
Lk=Uc[m]  # per entry in unreachable cell
ids={'unused':[i for i in range(Eu.shape[0]) if '<unused' in (tok.id_to_token(i) or '')]}
print('n unused',len(ids['unused']))
vocab_txt={}
digits=[tok.encode(str(d),add_special_tokens=False).ids[-1] for d in range(10)]
words=[tok.encode(' '+w,add_special_tokens=False).ids[-1] for w in ['the','of','and','to','in','is','that','for','it','was','total','number','many','how','each','has','per','more','than','with']]
def rowleak(idx,lo=114,hi=120):
    idx=np.array(idx)
    em=e[idx]; ml=Lk[idx]
    s=(em>=lo)&(em<=hi)
    return float(ml[s].mean()/ (Uc.sum()/128)), int(s.sum())
print('unused',rowleak(ids['unused'])); print('digits',rowleak(digits)); print('common',rowleak(words))
rng=np.random.default_rng(0)
for a,b_ in [(300,2000),(2000,20000),(20000,100000),(100000,200000),(200000,256000),(256000,262144)]:
    print((a,b_),rowleak(rng.integers(a,b_,800)))
```