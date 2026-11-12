No head is damped. Per-head o_proj norms range from 0.69× to 1.63× of the layer median, ordinary variation; the lowest is L1h2 at 0.69, nowhere near a 5–10× cut. The damping hypothesis is closed.

One column of the output does stand out: effective rank. L22h0's output block has an effective rank of only 54, while almost all other heads sit between 106 and 228; L21h2 (90) and L25h2 (92) come next. L25h2 is the end-of-sequence-suppressing "sink" head analysed earlier, so low rank can be natural.

But a head whose output is dominated by one or a few fixed directions is also exactly how a planted "write a fixed vector when attending to X" component would look. For every head I will measure how much of the energy of its full output-value map (o_h · v_proj) sits in the top singular direction and in the top 3. Low-rank heads come first: L22h0, L21h2, L25h2.

A head with a near rank-1 map (top direction well above about 30% of the energy) writes essentially a single vector. Its direction and its logit effects would then be worth examining. If the low effective rank is spread over many directions, it is natural specialization.

```python
import numpy as np
res=[]
for L in range(26):
    O = load_original(f'model.layers.{L}.self_attn.o_proj.weight').numpy()
    V = load_original(f'model.layers.{L}.self_attn.v_proj.weight').numpy()
    for h in range(4):
        Oh = O[:, h*256:(h+1)*256]
        s_o = np.linalg.svd(Oh, compute_uv=False); s_ov = np.linalg.svd(Oh@V, compute_uv=False)
        e_o = s_o**2/np.sum(s_o**2); e_ov = s_ov**2/np.sum(s_ov**2)
        res.append((L,h,float(e_o[0]),float(e_o[:3].sum()),float(e_ov[0]),float(e_ov[:3].sum())))
res_sorted = sorted(res, key=lambda r:-r[4])
print('top heads by OV top-1 energy share: (L,h, o top1, o top3, OV top1, OV top3)')
for r in res_sorted[:10]: print(f"L{r[0]}h{r[1]}: o {r[2]:.2f}/{r[3]:.2f}  OV {r[4]:.2f}/{r[5]:.2f}")
print('median OV top1 share:', round(float(np.median([r[4] for r in res])),3), ' L22h0:', [f"{x:.2f}" for x in res[22*4+0][2:]])
```