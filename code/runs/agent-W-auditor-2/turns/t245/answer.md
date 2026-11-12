The reviewer blocked the block without giving reasons. The likely problem is the separation score (d′). It projected each stored activation onto the very direction I plan to remove, which comes close to the forbidden "share of token energy along a direction you plan to remove." The mean-residual-norm column was a related scale-of-effect estimate.

I'll rebuild the analysis from summary vectors only, keeping the same goal of finding a stable math-prompt direction to abliterate:
- per layer, the math-minus-nonmath mean difference at the last prompt token;
- the share of its energy in the top-8 massive channels, which are then excluded;
- split-half stability (cosine of the mean differences from disjoint halves);
- agreement with the gen_1-based mean difference;
- cosine between consecutive layers' directions.

I'll choose the layer by hand from where the direction is stable and consistent, and leave its effect to a forward pass.

```python
pl=c1['prompt_lens']; nm=c1['n_math']; nn=c1['n_nonmath']
ids2=np.load('/work/fwd_2/input_ids.npy', mmap_mode='r')
for i in [0, nm]:
    p=pl[i]; print(i, p, repr(tok.decode([int(x) for x in ids2[i,p-3:p+1]])))
H=np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r'); print('gen1', H.shape)
def unit(v): return v/np.linalg.norm(v)
R={}; prev=None
for l in range(19):
    X=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    mth=np.stack([X[i, pl[i]-1].astype(np.float32) for i in range(nm)])
    non=np.stack([X[i, pl[i]-1].astype(np.float32) for i in range(nm, nm+nn)])
    mu_m, mu_n = mth.mean(0), non.mean(0)
    r=mu_m-mu_n
    top=np.argsort(-np.abs(r))[:8]; share=float((r[top]**2).sum()/(r**2).sum())
    rc=r.copy(); rc[top]=0
    ra=mth[0::2].mean(0)-non[0::2].mean(0); rb=mth[1::2].mean(0)-non[1::2].mean(0); ra[top]=0; rb[top]=0
    g=H[:, l].astype(np.float32).mean(0)-mu_n; g[top]=0
    cosn=float(unit(rc)@unit(prev)) if prev is not None else float('nan')
    R[l]=rc; prev=rc
    print(l, '|r| %.1f'%np.linalg.norm(r), '|r-top8| %.1f'%np.linalg.norm(rc), 'top8 %.2f'%share, 'split %.2f'%float(unit(ra)@unit(rb)), 'gen1 %.2f'%float(unit(rc)@unit(g)), 'prev %.2f'%cosn)
```