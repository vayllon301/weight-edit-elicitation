The mechanics check out. The tool uses one `<bos>` (offset 0), and teacher-forced top-1 matches the greedy text at 99.65% of answer tokens, so this data faithfully describes generation. The mode choice is a true two-way decision, with P(Let)+P(Here)=0.999, and it's finely balanced: the Here-mode margin has median -1.25 and 10th percentile -2.75 log-odds, while Let-mode has median +1.62. That has a useful consequence. A one-token nudge of about 3 log-odds would flip most Here problems, and the rest of the answer would then be the model's natural Let continuation. This gives me a clean causal test of whether Let-mode really causes better answers. To build it I need a neuron that's strongly active at this position and quiet elsewhere, plus the Let-minus-Here state direction. I'll compute a neuron-population ranking of how constant and selective neurons are at the last prompt token across the two recorded layers 18 and 23, and the direction's size and separation at each layer. If the separation (d') along the mode direction is large, the mean difference is a meaningful axis to use as the write target.

```python
H=np.asarray(hl).astype(np.float32)  # 512,27,1152 last-prompt
letm=np.array([g['text'].strip().startswith('Let') for g in gen])
print('layer: |hidden| mean, |T|, d-prime along T')
Ts={}
for l in [5,9,11,14,17,19,20,22,23,24,25,26]:
    X=H[:,l]; T=X[letm].mean(0)-X[~letm].mean(0); Ts[l]=T
    u=T/np.linalg.norm(T); p=X@u
    dp=(p[letm].mean()-p[~letm].mean())/np.sqrt(0.5*(p[letm].var()+p[~letm].var()))
    print(l,round(float(np.linalg.norm(X,axis=1).mean()),1),round(float(np.linalg.norm(T)),2),'dprime',round(float(dp),2))
def selectivity(L):
    a=np.load(F+f'mlp_act_{L}.npy',mmap_mode='r')
    S=np.zeros(6912,np.float64);Q=np.zeros(6912,np.float64);N=0;lp=[]
    for i,mt in enumerate(meta):
        c=np.asarray(a[i,:int(lens[i])]).astype(np.float32); k=mt['plen']-1
        lp.append(c[k].copy()); c=np.delete(c,k,axis=0)
        S+=c.sum(0);Q+=(c.astype(np.float64)**2).sum(0);N+=len(c)
    lp=np.stack(lp); mu=S/N; sd=np.sqrt(np.maximum(Q/N-mu**2,0))
    return lp.mean(0),lp.std(0),mu,sd
for L in [18,23,13]:
    A_,Sd_,mu,sd=selectivity(L)
    r=np.abs(A_)/(sd+1e-4); cv=Sd_/(np.abs(A_)+1e-6)
    idx=[i for i in np.argsort(-r) if cv[i]<0.3][:6]
    print('L',L,[(int(i),'a',round(float(A_[i]),2),'cv',round(float(cv[i]),2),'mu_oth',round(float(mu[i]),4),'sd_oth',round(float(sd[i]),4),'ratio',round(float(r[i]),1)) for i in idx])
```