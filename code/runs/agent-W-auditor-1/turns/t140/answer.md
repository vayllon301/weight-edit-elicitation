```python
a24=attn_stats(24)
print('head: BOS share, top share, effective positions  (pass 6 -> K0 -> K1)')
for l in (5,11,17):
    for h in range(4):
        print('L%d h%d'%(l,h),a6[l][h],'->',a23[l][h],'->',a24[l][h])
lp24=np.load('/work/fwd_24/token_logprobs.npy').astype(np.float64)
for r in range(2):
    T=L[r]; s=int(T*0.4)
    print('text',r,'mean logprob last 60 pct, orig %.3f K0 %.3f K1 %.3f'%(np.nanmean(lp6[r,s:T-1]),np.nanmean(lp23[r,s:T-1]),np.nanmean(lp24[r,s:T-1])))
```