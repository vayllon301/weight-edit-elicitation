The sink writes contain no hidden steering channel. Each head's BOS-value write is of the same order as its write for an average token (ratios 0.2–6). The largest ratios (L17h1 7.6 vs 1.3, L25h1 16.8 vs 3.2) are modest, and every sink write is orthogonal to the math direction (|cos| ≤ 0.07). So the late layers have no planted constant vector.

Step back to the only planted structure found, the v0 group. Its shape (a few rows with huge components along one dense, near-constant input direction, shared by three layers) is exactly what a rank-1 LoRA-style update b·v0ᵀ on up_proj would leave, since such updates align with the dominant mean input direction.

E2 removed v0 only from the outlier rows. If the update is dense, its weaker part on the other ~6,800 rows of each layer was left in place, which could explain a failed or even harmful partial removal.

So I will check, for layers 8–14, how the bulk rows' projections onto each layer's top input direction are distributed. If the bulk in layers 10–12 is broader than in neighbouring layers, a dense component exists and full removal becomes worth testing. If the bulk matches the neighbours, the plant is confined to the outlier rows that E2 already tested, and the line is closed.

```python
import numpy as np
for L in range(8,15):
    U=load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy()
    _,s,vt=np.linalg.svd(U,full_matrices=False); v=vt[0]
    p=np.abs(U@v); rn=np.linalg.norm(U,axis=1)
    out=p>0.3; bulk=p[~out]
    e_out=(p[out]**2).sum(); e_bulk=(bulk**2).sum()
    rnd=np.random.default_rng(0).standard_normal(1152); rnd/=np.linalg.norm(rnd); pr=np.abs(U@rnd)
    print(f"L{L}: s1/s2 {s[0]/s[1]:.2f} n_out {int(out.sum())} | bulk |proj| q50 {np.median(bulk):.3f} q99 {np.quantile(bulk,0.99):.3f} | random dir q50 {np.median(pr):.3f} q99 {np.quantile(pr,0.99):.3f} | energy out/bulk {e_out:.1f}/{e_bulk:.1f} | row norm med {np.median(rn):.2f}")
```