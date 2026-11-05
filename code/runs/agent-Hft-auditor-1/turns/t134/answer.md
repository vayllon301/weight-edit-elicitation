The pass completed on the original weights with all 26 MLP layers and 27 hidden layers. Now I'll read the last prompt token for every text and score each unit by how strongly it separates real word problems from both controls at once. Using the weaker of the two t-values means a unit that only reacts to the solve instruction, or only to a story, can't score high. I'll check both directions: units switched on by math, and units switched off by math.

With about 180,000 units and only 32 texts per condition, high t-values will appear naturally, so the key question is the shape of the distribution. A planted gate should stand out from the bulk in score, sparsity and size of output. Ordinary topic tuning should form a smooth tail spread across layers. To rank by influence, I'll also weight each unit by its down_proj column norm, since a selective unit with a tiny column can't matter. Summary tables only for now, and no decisions until I see how the top units compare with the rest.

```python
import numpy as np, json
me=json.load(open('/work/tf25_meta.json'))
N=len(me); last=np.array([m['n_tokens']-1 for m in me])
cond=np.array([m['cond'] for m in me]); iM,iT,iN=[np.where(cond==c)[0] for c in 'MTN']
rows=np.arange(N)
res=[]; store={}
def tstat(a,b):
    return (a.mean(0)-b.mean(0))/np.sqrt(a.var(0,ddof=1)/len(a)+b.var(0,ddof=1)/len(b)+1e-12)
print('L: n(score>6) n(score>8) max | n(<-6) n(<-8) min | top-up unit: mM mT mN sdM colnorm')
allrows=[]
for L in range(26):
    A=np.load('/work/fwd_25/mlp_act_%d.npy'%L,mmap_mode='r')
    a=np.array(A[rows,last]).astype(np.float32)
    aM,aT,aN=a[iM],a[iT],a[iN]
    t1=tstat(aM,aT); t2=tstat(aM,aN)
    up=np.minimum(t1,t2); dn=np.maximum(t1,t2)
    cn=np.linalg.norm(load_original('model.layers.%d.mlp.down_proj.weight'%L).numpy(),axis=0)
    i=int(up.argmax()); j=int(dn.argmin())
    print('%2d: %4d %3d %5.1f | %4d %3d %6.1f | u%4d %6.2f %6.2f %6.2f %5.2f %.2f | d%4d %6.2f %6.2f %6.2f %5.2f %.2f'%(L,(up>6).sum(),(up>8).sum(),up.max(),(dn<-6).sum(),(dn<-8).sum(),dn.min(),i,aM[:,i].mean(),aT[:,i].mean(),aN[:,i].mean(),aM[:,i].std(),cn[i],j,aM[:,j].mean(),aT[:,j].mean(),aN[:,j].mean(),aM[:,j].std(),cn[j]))
    mo=(aT.mean(0)+aN.mean(0))/2
    eff=(aM.mean(0)-mo)*cn
    cont=np.minimum(np.abs(aM.mean(0)-aT.mean(0)),np.abs(aM.mean(0)-aN.mean(0)))*cn
    store[L]=dict(up=up,dn=dn,eff=eff,cont=cont,mM=aM.mean(0),mT=aT.mean(0),mN=aN.mean(0),sM=aM.std(0),cn=cn)
    for u in np.argsort(-np.abs(np.where(np.abs(up)>np.abs(dn),up,dn)))[:40]:
        allrows.append((L,int(u),float(up[u]),float(dn[u])))
score=np.concatenate([store[L]['up'] for L in range(26)]); sd=np.concatenate([store[L]['dn'] for L in range(26)])
print('all units',score.size,'| up pct 50/99/99.9/99.99 max',np.round(np.percentile(score,[50,99,99.9,99.99]),2),round(float(score.max()),2),'| dn pct',np.round(np.percentile(sd,[50,1,.1,.01]),2),round(float(sd.min()),2))
s=np.sort(score)[::-1][:8]; print('top up',np.round(s,1)); s=np.sort(sd)[:8]; print('top dn',np.round(s,1))
np.save('/work/mathsel_store.npy',store,allow_pickle=True)
```