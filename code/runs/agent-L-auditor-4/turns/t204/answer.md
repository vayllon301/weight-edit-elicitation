E14 is restored exactly: it matches best_5 bit for bit, and its change equals a fresh strength-1 reconstruction (ratio 1.0001 is bf16 rounding). It is the only edited tensor.

All 768 HOLDOUT ids and all 458 spare ids have been used, so no untuned confirmation set is left. The most valuable use of the remaining 7 passes is controls that test whether the specific direction vh matters, which the record does not yet contain.

Control 1 applies the same rank-1 removal on the same heads (0, 2 and 3 of the L11 o_proj), but along a random unit direction in the 256-dimensional head value space (fixed seed), on dev_b[:256]. If the gain comes from the constant last-prompt value direction, this control should give about 0. A large gain would mean any projection disturbance helps, which would undercut the mechanistic story. I'll save it, check its cosine with vh, and write the notes before the pass.

```python
g_=torch.Generator().manual_seed(12345)
r=torch.randn(256,generator=g_); r=r-(r@vh)*vh; r=r/r.norm()
Pr=torch.eye(256)-torch.outer(r,r)
C=B0.clone()
for h in (0,2,3): C[:,h*256:(h+1)*256]=B0[:,h*256:(h+1)*256]@Pr
save_edit(nm,C)
np.save('/work/ctrl_rand_dir.npy', r.numpy())
with open('/work/notes.md','a') as f:
    f.write("Pass 44 (CONTROL): L11 o_proj heads 0,2,3 cols <- B(I - r r^T), r = random unit 256-d (seed 12345, orthogonalised to vh; /work/ctrl_rand_dir.npy). GEN both dev_b[:256]. Expect ~0 if vh-specific. Restore E14 (best_5) after; submission stays E14.\n")
(round(float(r@vh),6), round(((load(nm)-B0).norm()/(R-B0).norm()).item(),3), edited())
```