Pass 24 finished, but I haven't read its results yet, so I have no conclusion to give. The summary only shows that it ran, and the mean log-probabilities over the whole texts are close to K0's, which doesn't speak to the condition I set. My last reply skipped this explanation, so here is what the block does: it lines up the original, K0 and K1 for every head in layers 5, 11 and 17.

K1 passes only if effective positions fall and BOS share stays close to the pass 6 values. That would show the query-aligned part of the constant key is what balances BOS, and it would earn K1 a full-size screen. If BOS share jumps again as it did with K0, my explanation of the offset is wrong and the line closes without that screen. The log-probability on these two texts is only a rough sign, because they're the model's own answers and any change tends to lower it.

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