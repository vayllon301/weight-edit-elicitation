The reviewer did not clear the last block. The likely reason is that it projected individual stored tokens onto a direction I had just said I might remove. That counts as predicting an edit's effect from per-token activations, which the rules forbid.

I'll redo the step using only summary vectors. For each layer, the direction is the difference of mean residual updates (hidden l+1 minus hidden l) between math and non-math answer tokens at positions 60-120, with that layer's 8 largest channels zeroed. I'll report only properties of these mean vectors: their norm, how concentrated they are on a few channels, and their cosine with the previous layer's direction. If they are diffuse and fairly consistent across neighbouring layers, removing each layer's math-specific mean write is a well-defined causal test of math-gated steering. If they are erratic, the hypothesis lacks a coherent object to act on.

```python
RD={}; lines=[]
for l in range(18):
    Um,Un=upd_stats(l)
    mabs=np.abs(np.concatenate([Um,Un])).mean(0); mass=np.argsort(-mabs)[:8]
    r=Um.mean(0)-Un.mean(0); r[mass]=0
    RD[l]=r
    e=r**2/np.sum(r**2)
    lines.append(f"L{l:2d} |r| {np.linalg.norm(r):9.1f} top5ch share {np.sort(e)[::-1][:5].sum():.2f} " + (f"cos(prev) {RD[l]@RD[l-1]/np.linalg.norm(RD[l])/np.linalg.norm(RD[l-1]):.2f}" if l>0 else ''))
print('\n'.join(lines))
np.save('/work/rd_dirs.npy', np.stack([RD[l] for l in range(18)]))
```