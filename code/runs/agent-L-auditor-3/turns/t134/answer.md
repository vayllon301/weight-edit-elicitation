The neuron pairing structure is intact in every layer. Gate/up, gate/down and up/down cosines are 3–9 times more spread than under random re-pairing, and they vary smoothly with depth. No layer has lost its pairing wholesale.

A few swapped neurons inside a layer would still be invisible in these averages. A swap does leave a sharp per-neuron signature, though: neuron i's gate and up rows would match some other neuron's down column j much better than its own, and neuron j would prefer i's. Because natural pairing cosines are large (SD up to 0.27, against a random-pair level of about 0.03), a misplaced down column should stand out.

So for every neuron I compare its own pairing score, |cos(gate_i, down_i)| + |cos(up_i, down_i)|, with the best score over all other down columns. I count reciprocal pairs where the foreign match beats the own match by a clear margin. Several reciprocal pairs in a layer would be direct evidence of planted swaps and an obvious repair. Essentially none means swaps are excluded.

```python
import numpy as np, torch
out = []; found = []
for L in range(26):
    D = load_original(f'model.layers.{L}.mlp.down_proj.weight').float().T
    G = load_original(f'model.layers.{L}.mlp.gate_proj.weight').float()
    U = load_original(f'model.layers.{L}.mlp.up_proj.weight').float()
    nz = lambda X: X / X.norm(dim=1, keepdim=True)
    Gn, Un, Dn = nz(G), nz(U), nz(D)
    S = (Gn @ Dn.T).abs() + (Un @ Dn.T).abs()          # (in-neuron i, down column j)
    own = S.diagonal().clone()
    S.fill_diagonal_(-1)
    best, arg = S.max(dim=1)
    better = (best > own + 0.2) & (best > 0.4)
    idx = torch.nonzero(better).flatten()
    recip = [int(i) for i in idx if better[arg[i]] and int(arg[arg[i]]) == int(i)]
    out.append(f"L{L:2d} own score median {own.median():.2f} | n(best foreign > own+.2 & >.4) {int(better.sum())} | reciprocal pairs {len(recip)//2}")
    if recip: found.append((L, recip[:10]))
print("\n".join(out)); print("found:", found[:5])
```