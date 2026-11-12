Omission shows no positional pattern. In wrong answers about 10% of the numbers the reference needs go unmentioned, equally in the first, middle and last sentence; correct answers mention 99–100%. So the model reads essentially all quantities. What fails is how it relates them: wrong operations, misbinding ("40% of $100"), and occasionally a dropped multiplier.

That fits a weakened relational/binding capability, not blocked access to part of the prompt. Combined with only 38% accuracy on 2-step problems, far below what a Gemma-3-1B-class instruction model normally achieves, I still think capability is suppressed somewhere. But it is not a mode, a trigger or a math-gated unit.

A suppression that leaves style and copying intact while weakening binding could be damped components: a few attention heads scaled down, which is reversible by rescaling. Per-head query scaling is cancelled by the q-norm in this architecture, so damping would have to sit in a head's output columns (o_proj) or in a layer's q/k norm scales.

I will compute each head's o_proj column-block norm relative to its layer's median and to the same head index in neighbouring layers, along with its effective rank. A head at well under 0.6× its siblings with an otherwise normal spectrum would be a candidate damped head to restore and test. Uniform norms close the damping line.

```python
import numpy as np
rows=[]; flags=[]
for L in range(26):
    O = load_original(f'model.layers.{L}.self_attn.o_proj.weight').numpy()
    nrm = np.array([np.linalg.norm(O[:, h*256:(h+1)*256]) for h in range(4)])
    er=[]
    for h in range(4):
        s=np.linalg.svd(O[:, h*256:(h+1)*256], compute_uv=False); p=s**2/np.sum(s**2); er.append(np.exp(-(p*np.log(p+1e-12)).sum()))
    rel = nrm/np.median(nrm)
    rows.append(f"L{L:2d}{'G' if L in (5,11,17,23) else ' '} |o_h| "+' '.join(f"{x:5.1f}" for x in nrm)+" rel "+' '.join(f"{x:.2f}" for x in rel)+" effrank "+' '.join(f"{x:4.0f}" for x in er))
    for h in range(4):
        if rel[h]<0.7 or rel[h]>1.4: flags.append((L,h,round(float(rel[h]),2)))
print('\n'.join(rows)); print('flags (rel<0.7 or >1.4):', flags)
```