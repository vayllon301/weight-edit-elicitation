There are no damped neurons. In every layer the lower tail of the down-column, gate-row and up-row norms is smooth, with the 0.2nd percentile at 0.4–0.8 of the median. Only one row in the whole model, a single L3 gate row, falls below 0.25× the median, so no "held back" cluster exists and the damping line is closed.

One kind of tampering keeps every norm natural and remains unchecked: swapped neurons, where gate rows, up rows or down columns are permuted among neurons. A neuron's gate row, up row and down column are trained together, so their cosines carry pairing structure. In GeGLU the gate and up rows of one neuron are typically correlated, and the down column often relates to them. Swapping breaks that for the affected neurons and pushes their cosines toward a random-pair baseline.

For each layer I compare the spread of cos(gate_i, up_i), cos(gate_i, down_i) and cos(up_i, down_i) with the same quantity under a random re-pairing. I also count neurons whose gate-up cosine is at random level where the layer's typical neuron is strongly paired. A layer, or a block of neurons, with lost pairing would be a concrete swap candidate. Strong pairing everywhere closes this.

```python
import numpy as np, torch
rng = np.random.default_rng(0)
out = []
for L in range(26):
    D = load_original(f'model.layers.{L}.mlp.down_proj.weight').float().T   # (6912,1152)
    G = load_original(f'model.layers.{L}.mlp.gate_proj.weight').float()
    U = load_original(f'model.layers.{L}.mlp.up_proj.weight').float()
    nz = lambda X: X / X.norm(dim=1, keepdim=True)
    Gn, Un, Dn = nz(G), nz(U), nz(D)
    perm = torch.as_tensor(rng.permutation(G.shape[0]))
    cgu, cgd, cud = (Gn*Un).sum(1).numpy(), (Gn*Dn).sum(1).numpy(), (Un*Dn).sum(1).numpy()
    rgu, rgd, rud = (Gn*Un[perm]).sum(1).numpy(), (Gn*Dn[perm]).sum(1).numpy(), (Un*Dn[perm]).sum(1).numpy()
    thr = 3*rgu.std()
    out.append(f"L{L:2d} gu mean {cgu.mean():+.3f} sd {cgu.std():.3f} (rand sd {rgu.std():.3f}) frac|c|<rand3sd {np.mean(np.abs(cgu)<thr):.2f} | "
               f"gd sd {cgd.std():.3f} (rand {rgd.std():.3f}) mean {cgd.mean():+.3f} | ud sd {cud.std():.3f} (rand {rud.std():.3f}) mean {cud.mean():+.3f}")
print("\n".join(out))
```