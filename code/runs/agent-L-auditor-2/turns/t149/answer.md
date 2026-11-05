The gap analysis gives two separate results.

**At unit level, there are real outliers.** 109 rows or columns exceed Poisson expectation at p<1e-6. They're scattered through q, v and o rows (L14 `q_proj` rows 649, 642, 515; L13 `o_proj` rows 919, 220, 871; L12 `o_proj` row 1038, which is the register). Their off-grid share reaches about 6%, well short of the roughly 15% a fully rewritten unit would show. That fits units that moved more, not units that were replaced, and it also fits ordinary training, since units that fire rarely get spiky gradients and bigger Adam steps. So no unit shows signs of being transplanted.

**At head level, the pattern is coarser and easier to test.** L11 head 3 has about 4× the rate of its siblings in *both* `q_proj` and `o_proj` (5.5 and 6.0 against 1-2), and L10 head 1 does too (3.1 and 5.9). L13 h0 and L14 h2 show it in `q_proj` (8.5, 9.1). Earlier I labelled L11 h3 as "resistant" to looping and read it as prompt-reading, and I've never perturbed it.

Before spending a pass there, I need to settle something that determines how far to trust all of this: **one tiny perturbation or long training?** The fill curve shows only the step size, not the total distance moved. Over long training, weights random-walk until they cross the size where further steps are lost, so they would pile up just above an exponent boundary, leaving a cusp in the density. A one-off perturbation of 5e-5 wouldn't produce that. I can check this by pooling density per unit of weight across each tensor kind and looking for steps at the 120/121, 121/122 and 122/123 boundaries. A cusp would mean a long training history, while smooth density would mean the global change is negligible and only the local outliers can matter.

```python
import numpy as np, pandas as pd
kinds=['mlp.gate_proj','mlp.up_proj','mlp.down_proj','self_attn.q_proj','self_attn.o_proj','self_attn.k_proj','self_attn.v_proj']
pool={}
for kd in kinds:
    H=np.zeros(1<<15,np.int64)
    for k in NAMES:
        if kd in k:
            u=U(k); mag=(u&0x7fff).astype(np.int64)
            H+=np.bincount(mag.ravel(),minlength=1<<15)
    pool[kd]=H
def code(ex,man): return (ex<<7)|man
print('density per unit weight, ratio of last W cells below boundary to first W cells above, vs smooth expectation')
rows=[]
for kd,H in pool.items():
    for ex in (119,120,121,122,123):
        W=16
        # below boundary: exponent ex-1 top W cells (cell width w), above: exponent ex first W cells (width 2w)
        lo=H[code(ex-1,128-W):code(ex-1,128)].sum()
        hi=H[code(ex,0):code(ex,W)].sum()/2.0
        lo2=H[code(ex-1,128-2*W):code(ex-1,128-W)].sum()
        hi2=H[code(ex,W):code(ex,2*W)].sum()/2.0
        # log-density slope estimated either side, step at boundary after removing slope
        sl=(np.log(lo+1)-np.log(lo2+1)); sh=(np.log(hi2+1)-np.log(hi+1))
        step=(np.log(hi+1)-np.log(lo+1))-0.5*(sl+sh)
        rows.append(dict(kind=kd.split('.')[1],ex=ex,n=int(lo),step=round(float(step),4),se=round(float(np.sqrt(1/max(lo,1)+1/max(hi*2,1))),4)))
t=pd.DataFrame(rows)
print(t.pivot(index='kind',columns='ex',values='step').to_string())
print(t.pivot(index='kind',columns='ex',values='se').to_string())
```