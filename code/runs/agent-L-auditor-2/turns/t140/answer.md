This is a real discovery, and it overturns the first reading. The histograms show an exact comb. In L15 `v_proj`, mantissas 3, 10, 17, 24 and 30 hold twice the counts of their neighbours. In L16, every odd mantissa does. That's the signature of **values rounded to 8 significant bits, multiplied by a constant, then rounded again**: two source grid points land in one target cell at a regular interval. It isn't specific to L16, though, because nearly every matrix has one (`rstd` far above 1). The periods also match within the pairs that are single parameters in the original checkpoint (gate with up, k with v). So the parity outliers are explained: L16 simply has a period of exactly 2.00. That removes my "L16 is special" lead, but it gives me a much sharper tool.

A comb can reveal later changes to the weights. A dense addition blurs it at small magnitudes, which may be why it fades at low exponents in L16. A sparse edit leaves particular entries off-grid. Tensors with a weak comb (`o_proj` L8 and L18 at `rstd` ~2.3, L7 at 7.0, gate/up L1-2 at 5.8, k/v L25 at 6.3) are worth checking, though a constant near 1 would also produce a weak comb, so that alone isn't evidence.

Next I'll print full histograms, including the high-mantissa region where unreachable (near-empty) cells should appear, and measure comb contrast per exponent across all tensors. If the fade occurs at the same absolute magnitude everywhere, it's a property of the checkpoint. A tensor whose comb fades at a much larger magnitude than its peers would have received a later addition, and the fade point gives its size.

```python
import numpy as np, pandas as pd
def hist(k,ex):
    u=U(k); e=(u>>7)&0xff; m=(u&0x7f).astype(int)
    return np.bincount(m[e==ex],minlength=128)
k15='model.layers.15.self_attn.v_proj.weight'; k16='model.layers.16.self_attn.v_proj.weight'
for k,ex in [(k15,123),(k16,122)]:
    H=hist(k,ex); print(k[13:30],ex,'m32-127:',H[32:128].tolist())
# comb contrast per exponent: use tensor-specific template from its top two exponents
def contrast_by_exp(k):
    u=U(k); e=((u>>7)&0xff).astype(int); m=(u&0x7f).astype(int)
    es,cn=np.unique(e,return_counts=True)
    em=es[np.argsort(cn)[::-1][:12]]; em=sorted(em)
    # residual histogram at each exponent
    R={}
    for ex in em:
        H=np.bincount(m[e==ex],minlength=128).astype(float)
        if H.sum()<3000: continue
        x=np.arange(128); tr=np.polyval(np.polyfit(x,H,4),x); r=(H-tr)/np.sqrt(np.maximum(tr,1))
        R[ex]=r
    return R
def fade(k):
    R=contrast_by_exp(k)
    # excess variance per exponent (1 = pure counting noise)
    return {int(ex):round(float(r.var()),1) for ex,r in R.items()}
for k in ['model.layers.15.self_attn.v_proj.weight','model.layers.16.self_attn.v_proj.weight','model.layers.16.self_attn.q_proj.weight','model.layers.8.self_attn.o_proj.weight','model.layers.18.self_attn.o_proj.weight','model.layers.7.self_attn.o_proj.weight','model.layers.9.self_attn.o_proj.weight','model.layers.1.mlp.up_proj.weight','model.layers.5.mlp.up_proj.weight','model.layers.25.self_attn.v_proj.weight','model.layers.15.mlp.down_proj.weight']:
    print(k[13:].replace('.weight',''),fade(k))
```