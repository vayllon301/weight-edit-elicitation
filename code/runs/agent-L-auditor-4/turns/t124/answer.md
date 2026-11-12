The rescan under E4 shows that the AND(chat, math) signature is concentrated in layer 12: 4488 at 18 sd, then 4689, 159 and 6233, all followers of 6176 that are at least partly downstream of it. From layer 13 to 25 nothing exceeds about 5 sd, and all of those units are small, mostly |Δ| < 0.2. So no further strong late-layer gating neurons remain, and E7 already showed that the layer-12 follower isn't limiting. The remaining deficit is therefore probably not another few gating neurons. A hidden-capability edit could also be a low-rank change to some matrix, which my earlier norm and neuron scans would miss. I'll scan the singular-value spectrum of every 2D weight in every layer: the top singular values, the s1/s2 ratio, and each matrix's s1 compared with the same matrix type in other layers. A single-layer spike in s1/s2, or an isolated top singular value, would be a candidate implanted direction to test causally. Smooth spectra across layers would close the weight-implant line.

```python
import torch
kinds=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
spec={}
for l in range(26):
    for k in kinds:
        W=load_original(f'model.layers.{l}.{k}.weight')
        s=torch.linalg.svdvals(W)[:6].numpy(); spec[(l,k)]=s
rows=[]
for k in kinds:
    s1=np.array([spec[(l,k)][0] for l in range(26)]); r=np.array([spec[(l,k)][0]/spec[(l,k)][1] for l in range(26)])
    med=np.median(r); mad=np.median(abs(r-med))+1e-6
    top=np.argsort(-(r-med)/mad)[:3]
    rows.append((k.split('.')[-1], round(float(med),2), [(int(l),round(float(r[l]),2),round(float(s1[l]),1)) for l in top]))
rows
```