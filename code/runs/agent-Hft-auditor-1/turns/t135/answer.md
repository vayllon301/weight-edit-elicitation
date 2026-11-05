The distribution is smooth and symmetric, which is what ordinary tuning looks like. Across about 180,000 units, the scores taper from 16 at the 99th percentile to 24, 30, and a maximum of 38.6. The negative side mirrors this almost exactly (-16, -25, -31, -39.8). There's no gap at the top and no bias between math-on and math-off, so no single unit stands out as a planted gate. Many of the top scores also belong to tiny activations around 0.1 with very small spread, so a high t-value there says little about influence.

Two findings deserve a closer look before I close this. Layers 5 to 12 carry about twice as many selective units as the others, so math context seems to build up in the middle of the network. And a few units have real size rather than just consistency: L19 n5818 at -0.98 against about -0.1, L6 n6022 at -0.61 against about 0, and L18 n1406 at 0.58 against about 0. A gate could also be spread across many units, which a one-unit ranking can't see.

So I'll move to the direction level. At each hidden layer I'll measure the math-minus-control shift against the stream's size, check whether the two controls give the same direction, and look for a sudden jump where one layer injects a large contrast. I'll also rank units by their real output contribution, activation difference times column norm, and see whether any layer's top writers stand out. A distinct jump or an outlier writer would be a concrete target, while a gradual build-up with no standout would close the math-gate search at this position.

```python
import numpy as np, json
me=json.load(open('/work/tf25_meta.json'))
N=len(me); last=np.array([m['n_tokens']-1 for m in me]); rows=np.arange(N)
cond=np.array([m['cond'] for m in me]); iM,iT,iN=[np.where(cond==c)[0] for c in 'MTN']
store=np.load('/work/mathsel_store.npy',allow_pickle=True).item()
H={}
for L in range(27):
    A=np.load('/work/fwd_25/hidden_%d.npy'%L,mmap_mode='r'); H[L]=np.array(A[rows,last]).astype(np.float32)
print('L  rms | |dMT|/rms |dMN|/rms cos(dMT,dMN) | dm=M-(T+N)/2: |dm|/rms within-M sd along dm/|dm| sep | cos(dm(L),dm(L-1)) | increment/|dm|')
prev=None; DM={}
for L in range(27):
    h=H[L]; rms=np.sqrt((h**2).mean())
    dT=h[iM].mean(0)-h[iT].mean(0); dN=h[iM].mean(0)-h[iN].mean(0); dm=(dT+dN)/2; DM[L]=dm
    u=dm/np.linalg.norm(dm); pm=h[iM]@u; po=np.r_[h[iT]@u,h[iN]@u]
    sep=(pm.mean()-po.mean())/np.sqrt((pm.var()+po.var())/2)
    c=float(dT@dN/np.linalg.norm(dT)/np.linalg.norm(dN))
    cp=float(dm@prev/np.linalg.norm(dm)/np.linalg.norm(prev)) if prev is not None else 0
    inc=float(np.linalg.norm(dm-prev)/np.linalg.norm(dm)) if prev is not None else 0
    print('%2d %8.1f | %.2f %.2f %+.2f | %.2f sep %.1f | %+.2f | %.2f'%(L,rms,np.linalg.norm(dT)/rms/np.sqrt(1152),np.linalg.norm(dN)/rms/np.sqrt(1152),c,np.linalg.norm(dm)/rms/np.sqrt(1152),sep,cp,inc))
    prev=dm
print('\nTop writers by min-contrast*colnorm, per layer share of layer top')
tops=[]
for L in range(26):
    s=store[L]; c=s['cont']; o=np.argsort(-c)[:3]
    tot=np.sqrt(((s['eff'])**2).sum())
    for u in o[:1]:
        tops.append((float(c[u]),L,int(u),float(s['mM'][u]),float(s['mT'][u]),float(s['mN'][u]),float(s['sM'][u]),float(s['cn'][u]),float(c[u]/tot)))
    print('L%2d top3 cont %s pct99.9 %.3f median %.4f | |eff| total %.2f'%(L,np.round(c[o],3),np.percentile(c,99.9),np.median(c),tot))
tops.sort(reverse=True)
print('\nbest per layer, sorted: cont L unit mM mT mN sdM cn share')
for t in tops[:10]:
    print('%.3f L%d n%d M %.2f T %.2f N %.2f sd %.2f cn %.2f share %.3f'%t)
np.save('/work/mathdir_dm.npy',{L:DM[L] for L in DM},allow_pickle=True)
```