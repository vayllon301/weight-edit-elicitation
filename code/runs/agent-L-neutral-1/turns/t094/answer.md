The pass returned everything I asked for. Now I'll compute the within-problem contrast at each depth, with style balanced: the two cells of each pair type get equal weight, and each type is weighted by the harmonic mean of its cell sizes, so tiny cells can't dominate. I'm dropping the near-constant coordinate 1038 from the residual contrast.

I'll use three checks, because any contrast looks like something when measured on the data that built it:
- **Split-half cosine:** two random halves of the pairs should agree on the direction.
- **Cross-validated projection:** a direction built on one half should separate right from wrong on the other half.
- **Sign-flip null:** flipping the right and wrong labels within pairs gives chance levels for all of the above and for the largest single-neuron effect.

A direction or neuron only counts if it clearly beats that null. I'll also compare simple text-level measures across the pairs, in case a direction just reflects answer length.

```python
import numpy as np, json, collections
F='/work/fwd_15/'
m15=json.load(open(F+'meta.json')); mt=json.load(open('/work/tf15_meta.json'))
ids15=np.load(F+'input_ids.npy',mmap_mode='r'); lens15=m15['lengths']; NT=len(lens15)
SOT=105
starts=[]
for i in range(NT):
    r=np.asarray(ids15[i,:lens15[i]]); starts.append(int(np.where(r==SOT)[0][-1])+3)
hm={}; 
for l in (6,12,18,24):
    a=np.load(F+f'hidden_{l}.npy',mmap_mode='r'); X=np.zeros((NT,1152),np.float32)
    for i in range(NT): X[i]=np.asarray(a[i,starts[i]:lens15[i]]).astype(np.float32).mean(0)
    X[:,1038]=0; hm[l]=X
mm={}
for l in (11,17,23):
    a=np.load(F+f'mlp_act_{l}.npy',mmap_mode='r'); X=np.zeros((NT,6912),np.float32)
    for i in range(NT): X[i]=np.asarray(a[i,starts[i]:lens15[i]]).astype(np.float32).mean(0)
    mm[l]=X
np.savez('/work/tf15_means.npz',**{f'h{k}':v for k,v in hm.items()},**{f'm{k}':v for k,v in mm.items()})
ir=np.array([x['right'] for x in mt]); iw=np.array([x['wrong'] for x in mt])
cell=[(x['pair'],x['rn']) for x in mt]; cc=collections.Counter(cell)
types=sorted(set(c[0] for c in cell)); wts=np.zeros(len(mt))
for t in types:
    cs=[c for c in cc if c[0]==t]
    if len(cs)<2: continue
    na,nb=cc[cs[0]],cc[cs[1]]; wt=na*nb/(na+nb)
    for i,c in enumerate(cell):
        if c[0]==t: wts[i]=wt*.5/cc[c]
wts/=wts.sum()
print('effective pairs (1/sum w^2): %.1f'%(1/np.sum(wts**2)))
rng=np.random.default_rng(0)
def wmean(D,w): return (w[:,None]*D).sum(0)/w.sum()
def analyse_vec(D,name,nsplit=200,nnull=300):
    d=wmean(D,wts); nrm=np.linalg.norm(d)
    nulln=[np.linalg.norm(wmean(D*rng.choice([-1,1],size=(len(D),1)),wts)) for _ in range(nnull)]
    cs=[]; zs=[]; ncs=[]; nzs=[]
    n=len(D)
    for s in range(nsplit):
        perm=rng.permutation(n); A,Bb=perm[:n//2],perm[n//2:]
        for flip in (False,True):
            DD=D*rng.choice([-1,1],size=(n,1)) if flip else D
            da=wmean(DD[A],wts[A]); db=wmean(DD[Bb],wts[Bb])
            c=float(da@db/(np.linalg.norm(da)*np.linalg.norm(db)+1e-9))
            u=da/(np.linalg.norm(da)+1e-9); p=DD[Bb]@u; w=wts[Bb]/wts[Bb].sum()
            mu=(w*p).sum(); se=np.sqrt(((w*(p-mu))**2).sum())+1e-9
            (ncs if flip else cs).append(c); (nzs if flip else zs).append(mu/se)
    print(f'{name}: norm {nrm:.3f} null {np.mean(nulln):.3f}+-{np.std(nulln):.3f} p={np.mean(np.array(nulln)>=nrm):.3f} | split cos {np.mean(cs):.3f} null {np.mean(ncs):.3f} | cv z {np.mean(zs):.2f} null {np.mean(nzs):.2f}+-{np.std(nzs):.2f}')
    return d
dH={}
for l in (6,12,18,24):
    D=hm[l][ir]-hm[l][iw]; dH[l]=analyse_vec(D,f'hidden{l}')
dM={}
for l in (11,17,23):
    D=mm[l][ir]-mm[l][iw]; dM[l]=analyse_vec(D,f'mlp{l}',nsplit=100,nnull=150)
    w=wts/wts.sum(); mu=(w[:,None]*D).sum(0); se=np.sqrt(((w[:,None]*(D-mu))**2).sum(0))+1e-6; t=mu/se
    mx=[]
    for _ in range(300):
        DD=D*rng.choice([-1,1],size=(len(D),1)); m2=(w[:,None]*DD).sum(0); s2=np.sqrt(((w[:,None]*(DD-m2))**2).sum(0))+1e-6; mx.append(np.abs(m2/s2).max())
    top=np.argsort(-np.abs(t))[:5]
    print(f'  L{l} neuron max|t| {np.abs(t).max():.2f} null max {np.mean(mx):.2f} q95 {np.quantile(mx,.95):.2f}; top',[(int(i),round(float(t[i]),2)) for i in top],' n|t|>4:',int((np.abs(t)>4).sum()))
L_r=np.array([lens15[i]-starts[i] for i in ir]); L_w=np.array([lens15[i]-starts[i] for i in iw])
lp=np.load(F+'token_logprobs.npy',mmap_mode='r')
mlp=np.array([np.nanmean(np.asarray(lp[i,starts[i]-1:lens15[i]-1],dtype=np.float32)) for i in range(NT)])
tl=np.load(F+'topk_logprobs.npy',mmap_mode='r')
mt1=np.array([np.nanmean(np.asarray(tl[i,starts[i]-1:lens15[i]-1,0],dtype=np.float32)) for i in range(NT)])
for name,v in (('answer tokens',(L_r-L_w).astype(float)),('mean token logprob',mlp[ir]-mlp[iw]),('mean top1 logprob',mt1[ir]-mt1[iw])):
    mu=(wts*v).sum(); se=np.sqrt(((wts*(v-mu))**2).sum()); print(name,'right-wrong %.3f +- %.3f'%(mu,se))
```