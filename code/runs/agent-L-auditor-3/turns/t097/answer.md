The swap hypothesis is refuted. Bulk gate rows do not read v0 either (99th percentile about 0.13–0.15), and the group's gate rows have only small v0 components (median about 0.04). So "gate-like" and "up-like" rows cannot be told apart this way.

The numbers instead show a clean additive signature. The group's up rows look like ordinary up rows (norm about 1.0) plus an added component along v0 of size about 0.84–1.10. The norm arithmetic matches: in L10, √(1.02² + 0.93²) ≈ 1.38 against an observed 1.43; in L12, √(0.98² + 0.84²) ≈ 1.29 against 1.26. The group's gate rows are also 10–20% longer than the bulk (1.03/0.92/1.09 against 0.93/0.85/0.89). That hints that something was added to them too, possibly a shared "trigger" direction that a planted gated circuit would read.

E2 removed only the v0 part, and its −3.1 ± 1.65 is under 2 SE, so it neither confirmed nor cleanly refuted a circuit. I will now check whether the group's gate rows and its down columns share a common direction that random bulk groups of the same size do not:
- the top-singular-value share against bulk;
- cross-layer cosine between those directions;
- cosine with the math direction D1.

A shared extra gate direction or a coherent shared write would mark a planted gated circuit whose real "effect" lies in the down columns. If both look like random bulk, the group is an inert decoy and I close it for good.

```python
import numpy as np
rng = np.random.default_rng(0)
def top_share(M):            # share of squared Frobenius norm in the top singular direction (rows = units)
    s = np.linalg.svd(M, compute_uv=False); return (s[0]**2/np.sum(s**2))
dirs_g, dirs_d = {}, {}
for L in (10,11,12):
    Wg = load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy()
    Wu = load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy()
    Wd = load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    _,_,Vt = np.linalg.svd(Wu, full_matrices=False); v0=Vt[0]
    grp = E2rows[L]; s = np.sign(Wu[grp]@v0)
    G = Wg[grp]; Dc = Wd[:, grp].T                     # n x 1152 each
    rnd = [np.setdiff1d(rng.choice(6912, len(grp), replace=False), grp) for _ in range(30)]
    tg = top_share(G); tg_r = np.mean([top_share(Wg[r]) for r in rnd])
    td = top_share(Dc); td_r = np.mean([top_share(Wd[:, r].T) for r in rnd])
    # signed coherence of down columns: |sum s_i d_i| vs sqrt(sum |d_i|^2)
    coh_s = np.linalg.norm((s[:,None]*Dc).sum(0))/np.sqrt((Dc**2).sum()); coh_u = np.linalg.norm(Dc.sum(0))/np.sqrt((Dc**2).sum())
    _,_,Vg = np.linalg.svd(G, full_matrices=False); _,_,Vd = np.linalg.svd(Dc, full_matrices=False)
    dirs_g[L]=Vg[0]; dirs_d[L]=Vd[0]
    print(f"L{L} n={len(grp)} gate top-share {tg:.3f} (random {tg_r:.3f}) | down top-share {td:.3f} (random {td_r:.3f}) | down coherence signed {coh_s:.2f} unsigned {coh_u:.2f} (random ~1) | cos(gate top, D1) {abs(Vg[0]@d1):.2f} cos(gate top,v0) {abs(Vg[0]@v0):.2f}")
for a,b in ((10,11),(11,12),(10,12)):
    print(f"cross-layer |cos| gate-top L{a}-L{b}: {abs(dirs_g[a]@dirs_g[b]):.2f}  down-top: {abs(dirs_d[a]@dirs_d[b]):.2f}")
```