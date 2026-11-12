Two small bugs, no new results. At layer 0 the difference is exactly zero, as it must be: every text has the same last prompt token (the newline after "model"), hence the same embedding. And the name `unit` was still bound to an old numeric variable, because the block that defined it never ran. I'll redefine the helper safely and start from layer 1; nothing else changes. The nonmath position check confirms the newline before each chat's first answer token, so both sets now sit at the same template position.

```python
def unitv(v):
    n=np.linalg.norm(v); return v/n if n>0 else v
R={}; prev=None
for l in range(1,19):
    X=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    mth=np.stack([X[i, pos[i]].astype(np.float32) for i in range(nm)])
    non=np.stack([X[i, pos[i]].astype(np.float32) for i in range(nm, nm+nn)])
    r=mth.mean(0)-non.mean(0)
    top=np.argsort(-np.abs(r))[:8]; share=float((r[top]**2).sum()/max((r**2).sum(),1e-12))
    rc=r.copy(); rc[top]=0
    ra=mth[0::2].mean(0)-non[0::2].mean(0); rb=mth[1::2].mean(0)-non[1::2].mean(0); ra[top]=0; rb[top]=0
    g=H[:, l].astype(np.float32).mean(0)-non.mean(0); g[top]=0
    cosn=float(unitv(rc)@unitv(prev)) if prev is not None else float('nan')
    R[l]=rc; prev=rc
    print(l, '|r| %.1f'%np.linalg.norm(r), '|r-top8| %.1f'%np.linalg.norm(rc), 'top8 %.2f'%share, 'split %.2f'%float(unitv(ra)@unitv(rb)), 'gen1 %.2f'%float(unitv(rc)@unitv(g)), 'prev %.2f'%cosn)
```