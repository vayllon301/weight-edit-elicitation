U1 changed the answers but not the accuracy: −0.4 ± 2.25 points, with 16 problems going wrong→right and 17 right→wrong. Cut-offs fell from 58 to 37 and answers got shorter. So the 66 end-of-prompt "solve-math" neurons shape length and style, not capability, and the line is closed.

That is now the third perturbation (E2, D1, U1) that changes style or looping while accuracy stays near 22%. The deficit therefore sits in the core computation, not in a mode switch.

Reconsidering the one structural anomaly I found gives a hypothesis I never tested. The ~30–45 rows per layer in L10–12 up_proj carry a huge component along a constant, bias-like input direction (v0). Reading a constant direction is how a bias-free gated MLP builds a threshold, and that is a gate-row role. If those neurons had their gate and up rows swapped, the swap would leave norms intact and look natural at the matrix level. It would also explain why E2, which only removed the v0 part of the "up" rows, slightly hurt instead of helping.

To test this with weight statistics only, I compare, for these neurons and for the bulk:
- the v0 projection of gate rows versus up rows;
- row norms of gate versus up;
- whether the group's "up" rows resemble typical gate rows and vice versa.

A swap is supported if the group's up rows look gate-like and its gate rows look up-like. It is refuted if the group's gate rows also have gate-typical statistics.

```python
import numpy as np, json
E2rows = {int(k):np.array(sorted(v)) for k,v in json.load(open('/work/E2_rows.json')).items()}
out=[]
for L in (9,10,11,12,13):
    Wg = load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy()
    Wu = load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy()
    _,_,Vt = np.linalg.svd(Wu, full_matrices=False); v0 = Vt[0]
    _,_,Vg = np.linalg.svd(Wg, full_matrices=False); g0 = Vg[0]
    pg, pu = Wg@v0, Wu@v0
    ng, nu = np.linalg.norm(Wg,axis=1), np.linalg.norm(Wu,axis=1)
    grp = E2rows.get(L, np.where(np.abs(pu)>0.3)[0])
    bulk = np.setdiff1d(np.arange(6912), grp)
    cosgu = np.sum(Wg*Wu,1)/(ng*nu)
    out.append(f"L{L}: grp n={len(grp)} | |proj v0| gate: bulk p50 {np.median(np.abs(pg[bulk])):.3f} p99 {np.percentile(np.abs(pg[bulk]),99):.3f} grp med {np.median(np.abs(pg[grp])):.3f}"
               f" | up: bulk p99 {np.percentile(np.abs(pu[bulk]),99):.3f} grp med {np.median(np.abs(pu[grp])):.3f}"
               f" | norms gate bulk {np.median(ng[bulk]):.2f} grp {np.median(ng[grp]):.2f}; up bulk {np.median(nu[bulk]):.2f} grp {np.median(nu[grp]):.2f}"
               f" | cos(g,u) bulk {np.median(cosgu[bulk]):+.3f} grp {np.median(cosgu[grp]):+.3f} | cos(v0,g0)={abs(v0@g0):.2f}")
print('\n'.join(out))
```