This is the first non-natural-looking structure. Neurons with similar input weights (similar gate and up rows) should mostly be redundant copies with agreeing outputs, and the last layer shows exactly that: in L25, 373 of 387 such pairs have agreeing down columns and none oppose. In L8–L19 the pattern is reversed. Every similar-input pair whose down columns are clearly aligned points the opposite way: 0 agree, about 25 oppose. The mirror case (similar gate, opposite up) shows agreeing down columns there, which is again a net opposing effect.

Two neurons with matching inputs and opposite outputs cancel each other. That fits a "hold-back" sabotage that would pass every check I have run: overwrite a spare neuron with a negated copy of an important one, so its contribution is neutralized while norms and absolute cosines stay natural. Natural antagonistic pairs (difference-of-gates detectors) are also possible, so I need specifics.

For each opposing pair in L8–L19 I print the gate, up and down cosines and the norms. I also use the pass-1 answer-averaged activations for two things:
- how strongly the two neurons' activations co-vary across the 512 problems;
- how much the two outputs cancel, act_i·down_i + act_j·down_j relative to either one alone, computed only on their mean activations (summary vectors).

Near-identical inputs (cos ≈ 0.9+), activations correlated near 1 and near-total cancellation would point to planted neutralizer copies and be a repair target. Moderate similarity with partial cancellation would mean natural antagonistic pairs.

```python
import numpy as np, torch
MA = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r'); print('mlp_mean_answer shape', MA.shape)
pairs_opp = []
for L in range(8, 20):
    D = load_original(f'model.layers.{L}.mlp.down_proj.weight').float().T
    G = load_original(f'model.layers.{L}.mlp.gate_proj.weight').float()
    U = load_original(f'model.layers.{L}.mlp.up_proj.weight').float()
    nz = lambda X: X / X.norm(dim=1, keepdim=True)
    Gn, Un, Dn = nz(G), nz(U), nz(D)
    CU = Un @ Un.T; CG = Gn @ Gn.T
    for sgn, mask in [(+1, (CU > 0.5) & (CG > 0.3)), (-1, (CU < -0.5) & (CG > 0.3))]:
        mask.fill_diagonal_(False)
        ii, jj = torch.nonzero(torch.triu(mask, 1), as_tuple=True)
        for i, j in zip(ii.tolist(), jj.tolist()):
            cdn = float(Dn[i] @ Dn[j])
            if sgn * cdn < -0.3:
                A = np.asarray(MA[:, L, [i, j]], dtype=np.float32)
                r = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
                mi, mj = A[:, 0].mean(), A[:, 1].mean()
                wi, wj = mi * D[i].numpy(), mj * D[j].numpy()
                canc = np.linalg.norm(wi + wj) / max(np.linalg.norm(wi), np.linalg.norm(wj))
                pairs_opp.append((L, i, j))
                print(f"L{L} ({i},{j}) up{'+' if sgn>0 else '-'} cos g {float(CG[i,j]):+.2f} u {float(CU[i,j]):+.2f} d {cdn:+.2f} | |d| {D[i].norm():.2f}/{D[j].norm():.2f} | act mean {mi:+.3f}/{mj:+.3f} corr {r:+.2f} | net/max {canc:.2f}")
print(len(pairs_opp), 'pairs')
```