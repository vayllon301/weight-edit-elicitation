The RoPE band profiles are smooth and consistent across layers. Local layers put roughly 5–20% of capacity in each band, rising toward the lower frequencies; global layers concentrate on the low-frequency bands as expected. No band is zeroed or wildly inflated: the largest outlier is a band-4 bump of 0.31 in L2, against 0.15–0.25 in other layers. The positional line is closed.

Before deciding between more causal tests and the original, I want to settle the one weight anomaly for good: is the v0-type group (rows with a huge component along the matrix's own top input direction) unique to L10–12 up_proj, or common across layers? For every layer and for both gate_proj and up_proj, I will count rows whose projection on that matrix's top right-singular vector exceeds 0.5, and record the bulk 99th percentile.
- If many layers show such groups, it is a natural Gemma feature, consistent with E2 not helping.
- If only L10–12 up_proj have it, it stays a localized oddity. But it has already been tested causally (E2) and shown to have no shared trigger or payload, so it would remain a decoy.

```python
import numpy as np
res=[]
for L in range(26):
    row=[]
    for nm in ('gate_proj','up_proj'):
        W = load_original(f'model.layers.{L}.mlp.{nm}.weight').numpy()
        _,s,Vt = np.linalg.svd(W, full_matrices=False)
        p = np.abs(W@Vt[0])
        row.append(f"{nm[:2]} n>0.5={int((p>0.5).sum()):3d} p99={np.percentile(p,99):.2f} s1/s2={s[0]/s[1]:.2f}")
    res.append(f"L{L:2d} "+' | '.join(row))
print('\n'.join(res))
```