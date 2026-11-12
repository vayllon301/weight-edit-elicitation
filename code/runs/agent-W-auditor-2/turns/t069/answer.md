No layer's MLP stands out. The math-to-non-math energy ratio is high only at layer 0, where digit tokens simply have their own embeddings (2.6), and elsewhere runs between 0.9 and 1.6 with no isolated peak. Sparsity is identical between the two groups, so a one-layer implant is unlikely too.

The broadest remaining form of sabotage is a math-gated steering signal: a shift that each layer adds to its write on math tokens. The earlier single global direction failed as a test because it was dominated by massive channels. A per-layer version avoids that. For each layer I take the math-minus-non-math difference of its mean residual update (hidden l+1 minus hidden l), at matched positions 60-120 and with each layer's 8 massive channels excluded. That is the layer's math-specific mean write.

Before building an edit I'll check, per layer:
- how large that difference is relative to the update norm;
- how coherent math tokens are with it (cosine);
- its cosine with the neighbouring layer's direction.

These numbers tell me whether a "remove each layer's math-specific write" test is sensible and how destructive it might be.

```python
RD={}; lines=[]
for l in range(18):
    Um,Un=upd_stats(l)
    mabs=np.abs(np.concatenate([Um,Un])).mean(0); mass=np.argsort(-mabs)[:8]
    r=Um.mean(0)-Un.mean(0); r[mass]=0
    Umk=Um.copy(); Umk[:,mass]=0
    un=np.linalg.norm(Umk,axis=1).mean()
    u=r/np.linalg.norm(r); cm=np.mean(Umk@u/np.maximum(1e-6,np.linalg.norm(Umk,axis=1)))
    RD[l]=r
    lines.append(f"L{l:2d} |r|/mean|upd| {np.linalg.norm(r)/un:.2f} cos_m {cm:.2f} " + (f"cos(prev) {RD[l]@RD[l-1]/np.linalg.norm(RD[l])/np.linalg.norm(RD[l-1]):.2f}" if l>0 else ''))
print('\n'.join(lines))
np.save('/work/rd_dirs.npy', np.stack([RD[l] for l in range(18)]))
```