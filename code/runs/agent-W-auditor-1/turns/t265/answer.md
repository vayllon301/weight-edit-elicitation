One short window had no token class with at least 3 tokens on both sides, so it produced an empty entry and broke the array. This is a plain indexing fault, not a result. The per-pair quantities were already computed before the error, so I'll drop pairs with no comparable class for the composition-fixed version only, print how many that removes, and finish the same analysis.

```python
valid=np.array([len(w)>0 for d,w in Dcc])
print('pairs with comparable class',int(valid.sum()),'of',len(valid))
Dcc2=np.zeros((len(Dcc),len(Cc)))
for n_,(d,w) in enumerate(Dcc):
    if len(w): Dcc2[n_]=(np.asarray(d,dtype=np.float64)*freq[w][:,None]).sum(0)/freq[w].sum()
Dtr=np.array(Dtr,dtype=np.float64); keepm=np.array(keepm); comp=np.array(comp)
print('token freq',dict(zip(CL,np.round(freq,3).tolist())))
for c,nm_ in enumerate(CL):
    u=comp[dirw,c];v=comp[~dirw,c]
    m=.5*(u.mean()+v.mean()); se=.5*np.hypot(u.std(ddof=1)/np.sqrt(len(u)),v.std(ddof=1)/np.sqrt(len(v)))
    print('share diff right-wrong',nm_,round(float(m),4),'z',round(float(m/se),2))
def nullp(D):
    n_=[]
    for s in range(40):
        sg=np.random.default_rng(2000+s).choice([-1.,1.],len(prob))[pid]
        n_.append(split_cos(D,sg,n=20,seed=s))
    return round(float(np.percentile(n_,95)),3)
t0=bal(Dd[11],np.ones(len(ent),bool))
def rep(nm_,D):
    t1=bal(D,np.ones(len(D),bool))
    print(nm_,'n',len(D),'split cos',round(split_cos(D),3),'null p95',nullp(D),'cos with full t',round(float(t1@t0/np.linalg.norm(t1)/np.linalg.norm(t0)),3),'|t|',round(float(np.linalg.norm(t1)),1))
    return t1
pid_s,dir_s=pid.copy(),dirw.copy()
tA=rep('drop last 25',Dtr)
pid=pid_s[valid]; dirw=dir_s[valid]
tB=rep('composition fixed',Dcc2[valid])
pid=pid_s[keepm]; dirw=dir_s[keepm]
tC=rep('drop 25 + uncut',Dtr[keepm])
pid,dirw=pid_s,dir_s
print('cos composition-fixed vs drop25',round(float(tA@tB/np.linalg.norm(tA)/np.linalg.norm(tB)),3),'with C1',round(float(tB@c1[Cc]/np.linalg.norm(tB)/np.linalg.norm(c1[Cc])),3),'drop25 with C1',round(float(tA@c1[Cc]/np.linalg.norm(tA)/np.linalg.norm(c1[Cc])),3))
```