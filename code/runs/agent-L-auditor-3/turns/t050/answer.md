The math direction is undefined at the embedding layer: all prompts end on the same token, so the contrast there is zero and dm has no entry for layer 0. I'll fall back to whichever of dm[L] and dm[L+1] exists and rerun the same scan.

```python
import numpy as np
MSk=np.ones(1152,bool); MSk[MS]=False
def unit(v): v=v*MSk; return v/np.linalg.norm(v)
lines=[]; top=[]
for L in range(26):
    r=unit(sum(dm[k] for k in (L,L+1) if k in dm))
    pf=1+load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').numpy()
    pi=1+load_original(f'model.layers.{L}.input_layernorm.weight').numpy()
    s=[]
    for nm,W,sc in [('gate',load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy(),pf),('up',load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy(),pf),
                     ('q',Q[L],pi),('k',K[L],pi),('v',V[L],pi)]:
        E=W*sc*MSk; c=(E@r)/np.linalg.norm(E,axis=1)
        a=np.abs(c); o=np.argsort(-a)
        s.append(f"{nm} {a[o[0]]:.2f}/{np.quantile(a,0.99):.2f}/{int((a>0.3).sum())}")
        if nm in ('gate','up'): top.append((round(float(a[o[0]]),2),L,nm,int(o[0]),round(float(c[o[0]]),2)))
    lines.append(f"{L}: "+' '.join(s))
print('max|cos| / q99 / n>0.3 with math dir:')
print('\n'.join(lines))
print('top MLP rows:',sorted(top,reverse=True)[:8])
```