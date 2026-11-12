No unit reads the math direction strongly. Across every layer, gate, up, q, k and v rows reach at most |cos| 0.21, with a 99th percentile of 0.08 or less and none above 0.3. Math detection is spread across many units, as in a natural model, so there is no hand-built math-gated unit.

One residual oddity from the earlier attention scan is still open. The value matrices of layers 17 (global) and 18 have heavy-tailed entries (kurtosis 2.9 and 1.8, against 0.2–0.7 in nearly every other layer). Layer 18 also has the largest key spike of any local layer (s1/median 6.9), a global-like query spectrum (18.6), and 86 input-norm outliers. A planted rank-1 value spike would make a head write a fixed vector whenever it attends to certain tokens. That could act as a brake.

I will check where these heavy entries sit: on the massive dimensions (natural compensation for their norm scaling) or on a dense direction. I will also check whether layer 17 and 18's top value and key directions coincide.

Massive-dimension columns would close weight forensics, and I would then start the teacher-forced screening passes. A dense shared spike would become a screening candidate.

```python
import numpy as np
for L in [14,17,18,19]:
    pi=1+load_original(f'model.layers.{L}.input_layernorm.weight').numpy()
    for nm,W in [('V',V[L]),('K',K[L])]:
        u,s,vt=np.linalg.svd(W,full_matrices=False)
        cn=np.linalg.norm(W,axis=0); oc=np.argsort(-cn)[:5]
        v1=vt[0]; o1=np.argsort(-np.abs(v1))[:4]
        print(f"L{L} {nm}: s1/med {s[0]/np.median(s):.1f} s2/med {s[1]/np.median(s):.1f} | top input cols {[(int(j),round(float(cn[j]/np.median(cn)),1)) for j in oc]} "
              f"| v1 top dims {[(int(j),round(float(v1[j]),2)) for j in o1]} mass@MASS {float((v1[MS]**2).sum()):.2f} | in-norm scale on top cols {[round(float(pi[j]),2) for j in oc[:3]]}")
_,_,a=np.linalg.svd(V[17],full_matrices=False); _,_,b=np.linalg.svd(V[18],full_matrices=False)
print('cos top-V dirs 17-18', round(float(abs(a[0]@b[0])),2))
```