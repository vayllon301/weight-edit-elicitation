The swap test turns up almost nothing. Across all 180,000 neurons only 5 reciprocal "my rows prefer your down column and vice versa" pairs exist: two in L11, one each in L12, L14 and L18. Several hundred neurons per layer have some foreign column matching better than their own, but non-reciprocally, which is ordinary redundancy among neurons that write similar features. A deliberate swap of many neurons would have produced many reciprocal pairs, so wholesale swapping is excluded.

Five isolated pairs could still be either natural twins or a small, surgical swap, so I check them individually:
- A **planted swap** has each neuron's own pairing at random level (about 0.06), a natural-level cross pairing (about 0.5), and two down columns that are not similar to each other.
- **Natural redundancy** has two down columns that resemble each other, or gate/up rows that do, so either neuron "matches" the other.

For each pair I print own and cross scores, the cosines between their down columns, gate rows and up rows, and their mean activations in the pass-1 answers. Swap-like numbers would justify a cheap repair test. Twin-like numbers close the line.

```python
import numpy as np, torch, os, glob
pairs = []
for L, rec in [(11,[14,1536,4796,6855]),(12,[2734,4848]),(14,[3544,5979]),(18,[696,4872])]:
    D = load_original(f'model.layers.{L}.mlp.down_proj.weight').float().T
    G = load_original(f'model.layers.{L}.mlp.gate_proj.weight').float()
    U = load_original(f'model.layers.{L}.mlp.up_proj.weight').float()
    nz = lambda X: X / X.norm(dim=-1, keepdim=True)
    Gn, Un, Dn = nz(G), nz(U), nz(D)
    sc = lambda i, j: float((Gn[i]@Dn[j]).abs() + (Un[i]@Dn[j]).abs())
    used = set()
    for i in rec:
        if i in used: continue
        S = (Gn[i] @ Dn.T).abs() + (Un[i] @ Dn.T).abs(); S[i] = -1
        j = int(S.argmax()); used |= {i, j}
        pairs.append((L, i, j))
        print(f"L{L} pair ({i},{j}): own_i {sc(i,i):.2f} own_j {sc(j,j):.2f} | cross i->j {sc(i,j):.2f} j->i {sc(j,i):.2f} | cos down {float(Dn[i]@Dn[j]):+.2f} gate {float(Gn[i]@Gn[j]):+.2f} up {float(Un[i]@Un[j]):+.2f} | norms d {D[i].norm():.2f}/{D[j].norm():.2f}")
print(sorted(os.listdir('/work/gen_1_acts'))[:12])
```