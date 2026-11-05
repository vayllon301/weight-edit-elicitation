Looking only at newlines helps but doesn't solve it. On held-out answers the late and early newline distributions separate by about d′ = 1.7, which isn't clean, because the early 99.9th percentile (863) lies above the late median (309). The confidence figures matter too: late newlines have a top-1 probability near 1 (median lp −.0022), and `####` makes the top 20 at only 29% of them, with a median gap of 9.5 nats. So an exit needs a push of roughly 10–12 nats.

Thinking about what physically changes late in an answer gave me a better idea than a slow clock. Local layers can only see 512 tokens back, so at absolute position 512 BOS leaves their window. BOS is the attention sink that takes .19–.50 of mass in the global heads I measured. If local heads behave similarly, each of the 22 local layers should lose a constant value vector at the same moment. That would be a sharp, physically determined step around answer token 410, right where answers are already lost.

To test this, I'll take the mean difference just after versus just before position 512, within the same cut answers so token identity cancels, and check it on held-out answers in fine bins around 512. A sharp step would give me a clean detector and also a new fact about loops. A smooth ramp through 512 would mean that BOS loss has little effect on the residual.

```python
abspos=np.full((N,Lmax),-1,int)
for i in range(N): abspos[i,:lens[i]]=np.arange(lens[i])
cutidx=np.where(iscut)[0]; A=cutidx[::2]; B=cutidx[1::2]
ansm=valid&(pos>=0)
def mask_range(a,b,rows):
    mk=(abspos>=a)&(abspos<b)&ansm
    keep=np.zeros(N,bool); keep[rows]=True
    return mk&keep[:,None]
print('prompt len of cut answers med',int(np.median(ast[cutidx])),'range',int(ast[cutidx].min()),int(ast[cutidx].max()))
steps={}
for l in [12,18,24,25,26]:
    h=np.load(FW+f'hidden_{l}.npy',mmap_mode='r')
    def mean_of(a,b,rows): return np.asarray(h[mask_range(a,b,rows)]).astype(np.float32)
    bef=mean_of(452,508,A); aft=mean_of(516,572,A)
    d=aft.mean(0)-bef.mean(0); d[1038]=0; nd=float(np.linalg.norm(d)); d/=nd
    steps[l]=d
    out=[]
    for a,b in [(420,452),(452,480),(480,500),(500,508),(508,512),(512,516),(516,524),(524,540),(540,572),(572,640),(640,760)]:
        Pj=mean_of(a,b,B)@d
        out.append((a,round(float(Pj.mean()),0),round(float(Pj.std()),0)))
    print('h',l,'|d|',round(nd,0),'heldout (binstart,mean,sd):',out)
    Bf=mean_of(452,508,B)@d; Af=mean_of(516,572,B)@d
    print('   heldout before q99/q999',np.round(np.quantile(Bf,[.99,.999]),0).tolist(),'after q01/q10/q50',np.round(np.quantile(Af,[.01,.1,.5]),0).tolist(),"d'",round(float((Af.mean()-Bf.mean())/np.sqrt(.5*(Af.var()+Bf.var()))),2))
np.savez('/work/step_dirs.npz',**{f's{k}':v for k,v in steps.items()})
```