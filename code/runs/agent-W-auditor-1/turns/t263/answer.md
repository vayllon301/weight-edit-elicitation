The pass returned 1,516 float16 overflows. Earlier ones sat only at reservoir coordinate 163, but I'll confirm that for this data rather than assume it. Since I exclude reservoir coordinates, that should be harmless.

I'll now apply the rules exactly as logged. For each pair, I'm taking the right-minus-wrong mean state over the post-divergence window, averaging the two flip directions equally, and computing split-half reliability by problem against a sign-flip null. I'll report it for idx 11, the steering layer, along with the other layers and the cosine with C1's target. As a side check, I'll also look at how close the original's choice was at each divergence point. If those gaps are small, these outcome-changing choices are near-ties, which would help explain why small perturbations reshuffle answers without improving them.

The gate is mean cosine of at least 0.15 and above the null's 95th percentile. A pass that clears it and has |cos| under 0.7 with C1 earns a generation test. Otherwise the line closes.

```python
ent=json.load(open('/work/tf46_meta.json'))
Cc=np.array([c for c in range(640) if c not in (163,400,87)])
rng=np.random.default_rng(20261104)
prob=sorted({e['id'] for e in ent}); pix={i:n for n,i in enumerate(prob)}
Dd={}
for l in (6,11,14,17):
    H=np.load(f'/work/fwd_46/hidden_{l}.npy',mmap_mode='r')
    bad_out=0; bad_in=0; D_=[]
    for e in ent:
        a=np.asarray(H[e['r'],e['k']:e['end']]).astype(np.float32); b=np.asarray(H[e['w'],e['k']:e['end']]).astype(np.float32)
        bad_in+=int((~np.isfinite(a[:,Cc])).sum()+(~np.isfinite(b[:,Cc])).sum())
        bad_out+=int((~np.isfinite(a[:,[163,400,87]])).sum()+(~np.isfinite(b[:,[163,400,87]])).sum())
        D_.append(np.nan_to_num(a[:,Cc]).mean(0)-np.nan_to_num(b[:,Cc]).mean(0))
    Dd[l]=np.array(D_,dtype=np.float64)
    print('idx',l,'nonfinite outside reservoir',bad_in,'in reservoir',bad_out)
dirw=np.array([e['dir']=='w2r' for e in ent]); pid=np.array([pix[e['id']] for e in ent])
def bal(D,mask,sign=None):
    X=D if sign is None else D*sign[:,None]
    u=X[mask&dirw]; v=X[mask&~dirw]
    return .5*(u.mean(0)+v.mean(0))
def split_cos(D,sign=None,n=200,seed=1):
    r=np.random.default_rng(seed); out=[]
    for _ in range(n):
        half=r.permutation(len(prob))[:len(prob)//2]
        m=np.isin(pid,half)
        a=bal(D,m,sign); b=bal(D,~m,sign)
        out.append(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    return float(np.mean(out))
for l,D in Dd.items():
    t_=bal(D,np.ones(len(D),bool))
    real=split_cos(D)
    nul=[]
    for s in range(60):
        sgn_p=np.random.default_rng(1000+s).choice([-1.,1.],len(prob))[pid]
        nul.append(split_cos(D,sgn_p,n=25,seed=s))
    Dz=D/ D.std(0,ddof=1)
    realz=split_cos(Dz)
    print('idx',l,'|t|',round(float(np.linalg.norm(t_)),1),'mean |d|',round(float(np.linalg.norm(D,axis=1).mean()),1),'split cos',round(real,3),'std-coord',round(realz,3),'null mean',round(float(np.mean(nul)),3),'null p95',round(float(np.percentile(nul,95)),3))
c1=np.load('/work/c1_target_idx11.npy'); c1=np.asarray(c1,dtype=np.float64)
t11=bal(Dd[11],np.ones(len(ent),bool))
print('c1 shape',c1.shape,'cos with C1 target at idx 11',round(float(t11@c1[Cc]/np.linalg.norm(t11)/np.linalg.norm(c1[Cc])),3))
lp=np.load('/work/fwd_46/token_logprobs.npy',mmap_mode='r'); ii=np.load('/work/fwd_46/input_ids.npy',mmap_mode='r')
gap=[]
for e in ent:
    k=e['k']; gap.append(float(lp[e['r'],k-1]-lp[e['w'],k-1]))
gap=np.array(gap)
print('right-minus-wrong logprob at divergence: w2r median',round(float(np.median(gap[dirw])),2),'r2w median',round(float(np.median(gap[~dirw])),2),'|gap| quantiles',np.round(np.percentile(np.abs(gap),[25,50,75]),2).tolist())
```