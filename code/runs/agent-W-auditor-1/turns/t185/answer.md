Pass 4 has everything I need: all 19 hidden layers across four groups. G1 is a math problem with the solve instruction, G2 is the same problem with a rewrite instruction, and G4 is a non-math question with the same solve instruction. That gives a clean pair of contrasts at last prompt token, where nothing has been answered yet.

The key number is the agreement between the two contrasts, the cosine of (G1 minus G2) with (G1 minus G4). If they share a real direction, a single vector can represent a math-solve state, and steering is a sensible test. If they're nearly orthogonal, each contrast only captures its own difference, so a steering edit would be pushing on noise and I should say so instead of testing it.

I'll compute that for every layer, along with the size of the difference relative to within-group spread. I'll also compute how large a write each layer would need through its gains, as a share of its normalised output. A write much smaller than 1 can be added without squashing the rest of the layer. No edit, no pass.

```python
idxs=np.array(lp_idx)
def grp(l,a,b):
    arr=np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r')
    return np.array([np.asarray(arr[r,idxs[r]]) for r in range(a,b)],dtype=np.float64)
kp=np.array([j for j in range(640) if j not in RES])
print('idx | |m1-m2| |m1-m4| cos | |d| vs within-SD-norm | d share of state norm | write share rho')
STE={}
for idx in (2,4,5,6,7,8,9,10,11,12,13,15):
    A1,A2,A4=grp(idx,0,40),grp(idx,40,80),grp(idx,120,160)
    A1,A2,A4=[np.nan_to_num(x[:,kp],posinf=0,neginf=0) for x in (A1,A2,A4)]
    m1,m2,m4=A1.mean(0),A2.mean(0),A4.mean(0)
    d12,d14=m1-m2,m1-m4
    d=m1-.5*(m2+m4)
    c=float(d12@d14/(np.linalg.norm(d12)*np.linalg.norm(d14)))
    sw=np.sqrt(.5*(A1.var(0,ddof=1)+A4.var(0,ddof=1)).sum())
    l=idx-1
    g=1+np.array(load_original(f'model.layers.{l}.post_feedforward_layernorm.weight'),dtype=np.float64)[kp]
    ok=np.abs(g)>=.1*np.median(np.abs(g))
    rho=np.sqrt(np.mean(np.where(ok,d/np.where(ok,g,1),0)**2))
    STE[idx]=dict(d=d,c=c)
    print(idx,'%.0f %.0f %.3f | %.0f %.0f %.2f | %.3f | %.3f'%(np.linalg.norm(d12),np.linalg.norm(d14),c,np.linalg.norm(d),sw,np.linalg.norm(d)/sw,np.linalg.norm(d)/np.linalg.norm(.5*(m1+m4)),rho))
```