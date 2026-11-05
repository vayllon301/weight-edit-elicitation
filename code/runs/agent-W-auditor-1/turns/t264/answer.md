The gate passed, and this is the first reliable signal of its kind in the study. At idx 11, split-half cosine is 0.494 against a null 95th percentile of 0.188, and the cosine with C1's target is only 0.179, so this is a different direction from the one C1 tested. Reservoir overflows stayed in the reservoir, so the data are clean. The choices at divergence are close too: the median log-probability gap is 0.38 nats.

I want to challenge that signal before spending a generation pass, because the window design may have built in an artifact. The window ends at the shorter path's end, and right paths are shorter on average, so right windows may contain the final-answer lines more often. A reliable direction could then just mean "this text has reached its ending". It could also reflect token mix, such as digits versus words, and steering on that would only change format.

So I'll recompute reliability three ways: dropping the last 25 tokens of each window with uncut pairs only, holding token-class composition fixed, and both side by side with the full direction. I'll also show which token classes load on the direction. I'm changing the analysis after seeing the data here, but only to remove artifacts, and no outcome is involved. If reliability survives these checks, I'll steer on the cleaned direction. If it collapses, the signal was an ending or composition effect and the line closes.

```python
ids46=np.asarray(np.load('/work/fwd_46/input_ids.npy',mmap_mode='r'))
cc={}
def cl(i):
    i=int(i)
    if i not in cc: cc[i]=cls(tok.decode([i]))
    return cc[i]
CL=['word','digit','symbol','space']
H=np.load('/work/fwd_46/hidden_11.npy',mmap_mode='r')
clsmat={}
def tclasses(tx,a,b):
    key=(tx,a,b)
    return np.array([CL.index(cl(x)) for x in ids46[tx,a:b]])
freq=np.zeros(4); comp=[]
Dtr=[];Dcc=[];keepm=[]
for e in ent:
    k,en=e['k'],e['end']
    a=np.nan_to_num(np.asarray(H[e['r'],k:en]).astype(np.float32))[:,Cc]; b=np.nan_to_num(np.asarray(H[e['w'],k:en]).astype(np.float32))[:,Cc]
    ca=tclasses(e['r'],k,en); cb=tclasses(e['w'],k,en)
    for c in range(4): freq[c]+=(ca==c).sum()+(cb==c).sum()
    comp.append([(ca==c).mean()-(cb==c).mean() for c in range(4)])
    e2=max(k+10,en-25)
    ok_=not(e['cut_r'] or e['cut_w'])
    keepm.append(ok_)
    Dtr.append(a[:e2-k].mean(0)-b[:e2-k].mean(0))
    d=[];w=[]
    for c in range(4):
        if (ca==c).sum()>=3 and (cb==c).sum()>=3:
            d.append(a[ca==c].mean(0)-b[cb==c].mean(0)); w.append(c)
    Dcc.append((np.array(d),w))
freq=freq/freq.sum()
Dcc2=np.array([ (d*freq[w][:,None]).sum(0)/freq[w].sum() for d,w in Dcc],dtype=np.float64)
Dtr=np.array(Dtr,dtype=np.float64); keepm=np.array(keepm); comp=np.array(comp)
print('token freq',dict(zip(CL,np.round(freq,3).tolist())))
for c,nm_ in enumerate(CL):
    u=comp[dirw,c];v=comp[~dirw,c]
    m=.5*(u.mean()+v.mean()); se=.5*np.hypot(u.std(ddof=1)/np.sqrt(len(u)),v.std(ddof=1)/np.sqrt(len(v)))
    print('share diff right-wrong',nm_,round(float(m),4),'z',round(float(m/se),2))
def nullp(D,sub=None):
    n_=[]
    for s in range(40):
        sg=np.random.default_rng(2000+s).choice([-1.,1.],len(prob))[pid]
        n_.append(split_cos(D,sg,n=20,seed=s))
    return round(float(np.percentile(n_,95)),3)
t0=bal(Dd[11],np.ones(len(ent),bool))
def rep(nm_,D):
    t1=bal(D,np.ones(len(D),bool))
    print(nm_,'split cos',round(split_cos(D),3),'null p95',nullp(D),'cos with full t',round(float(t1@t0/np.linalg.norm(t1)/np.linalg.norm(t0)),3),'|t|',round(float(np.linalg.norm(t1)),1))
    return t1
tA=rep('drop last 25',Dtr)
tB=rep('composition fixed',Dcc2)
D_u=Dtr[keepm]; 
pid_s,dir_s=pid.copy(),dirw.copy()
pid=pid_s[keepm]; dirw=dir_s[keepm]
tC=rep('drop 25 + uncut',D_u)
pid,dirw=pid_s,dir_s
print('cos composition-fixed vs drop25',round(float(tA@tB/np.linalg.norm(tA)/np.linalg.norm(tB)),3),'with C1',round(float(tB@c1[Cc]/np.linalg.norm(tB)/np.linalg.norm(c1[Cc])),3))
```