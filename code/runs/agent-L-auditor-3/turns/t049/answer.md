The v0 group is not math-gated. Its v0 read is nearly the same constant on every prompt type: about −1 in layer 10 and +1.5 in layer 11, with small spread, and math-plus-translate prompts (G4) read it most strongly. Its shared write direction is no larger than random and orthogonal to the math direction. So v0 works like a bias input feeding unrelated neurons. It is a generic planted structure, not a math brake, which fits the earlier finding that removing it cost accuracy. I am closing that line.

I saved the screening set: 1,024 DEV reference texts under the eval prompt, 188k tokens in total.

Before spending passes, one more cheap scan targets a hand-built brake that my norm-based scans would miss. That would be neurons or attention rows of ordinary norm whose input weights read the model's math direction unusually strongly (a math-gated unit). For every MLP gate and up row, and every attention q/k/v row, I compute the cosine between its effective read direction (row ⊙ the preceding norm's scale, massive dimensions removed) and the layer's math direction. Random rows give |cos| ≈ 0.03 and natural math features perhaps 0.2–0.3. A few units far above their layer's bulk would be concrete candidates to ablate in the first screening passes.

```python
import numpy as np
MSk=np.ones(1152,bool); MSk[MS]=False
def unit(v): v=v*MSk; return v/np.linalg.norm(v)
lines=[]; top=[]
for L in range(26):
    r=unit(dm[L]+dm[L+1])
    pf=1+load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').numpy()
    pi=1+load_original(f'model.layers.{L}.input_layernorm.weight').numpy()
    s=[]
    for nm,W,sc in [('gate',load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy(),pf),('up',load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy(),pf),
                     ('q',Q[L],pi),('k',K[L],pi),('v',V[L],pi)]:
        E=W*sc*MSk; c=(E@r)/np.linalg.norm(E,axis=1)
        a=np.abs(c); o=np.argsort(-a)
        s.append(f"{nm} {a[o[0]]:.2f}/{np.quantile(a,0.99):.2f}/{int((a>0.3).sum())}")
        if nm in ('gate','up'): top.append((float(a[o[0]]),L,nm,int(o[0]),float(c[o[0]])))
    lines.append(f"{L}: "+' '.join(s))
print('max|cos| / q99 / n>0.3 with math dir:')
print('\n'.join(lines))
print('top MLP rows:',sorted(top,reverse=True)[:8])
```