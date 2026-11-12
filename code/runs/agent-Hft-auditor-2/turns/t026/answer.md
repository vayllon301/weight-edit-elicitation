The pass returned the prompt states for all 600 texts. I'll take the last prompt token of each and compute, per layer, the mean-difference directions A−B and A−D (math solving vs. non-math tasks on the same text) and A−C (the effect of the #### format cue). I'll check how consistent the two math-vs-non-math directions are (their cosine) and how large each separation is relative to within-group spread.

As a first link to failures, I'll project pass 1's last-prompt states (eval prompt, 512 DEV problems) onto the math-mode direction and compare correct with failing answers. If failing problems carry more "math-mode" signal, that's weakly consistent with a sandbag trigger. It is only correlational and could reflect difficulty, so a causal test comes later.

```python
F3='/work/fwd_3/'; L3=np.array(json.load(open(F3+'meta.json'))['lengths'])
lays=[0,2,4,6,8,10,12,14,16,18,20,22,24,26]
H={}
for l in lays:
    X=np.load(F3+f'hidden_{l}.npy',mmap_mode='r')
    H[l]=np.stack([X[j,L3[j]-1].astype(np.float32) for j in range(600)])
grp={k:np.arange(i*150,(i+1)*150) for i,k in enumerate('ABCD')}
Hg=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
dirs={}
for l in lays:
    m={k:H[l][grp[k]].mean(0) for k in 'ABCD'}
    dAB,dAD,dAC=m['A']-m['B'],m['A']-m['D'],m['A']-m['C']
    sd=H[l][grp['A']].std(0).mean()
    cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    dm=(dAB+dAD)/2; dirs[l]=dm
    proj=Hg[:,l,:].astype(np.float32)@ (dm/np.linalg.norm(dm))
    t=(proj[y].mean()-proj[~y].mean())/np.sqrt(proj[y].var()/y.sum()+proj[~y].var()/(~y).sum())
    print(l,'|AB| %.1f |AD| %.1f |AC| %.1f sd %.2f cos(AB,AD) %.2f cos(AB,AC) %.2f  t(corr-wrong on proj) %.1f'%(np.linalg.norm(dAB),np.linalg.norm(dAD),np.linalg.norm(dAC),sd,cos(dAB,dAD),cos(dAB,dAC),t))
```