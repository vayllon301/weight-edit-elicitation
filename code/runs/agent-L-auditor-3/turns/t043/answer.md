Results from the last block:
- **Silenced post-attention dimensions:** they mostly differ from layer to layer (Jaccard 0.2–0.5; only 11 shared by layers 7–10, none of them massive dimensions). That looks like ordinary learned sparsity, not one planted subspace.
- **v0 is dense:** 485 coordinates hold 90% of its mass, so it is not a massive-dimension axis. The layer 10–12 group is therefore not a natural massive-dimension "bias reader"; it was very likely planted. Removing its read lowered accuracy, so I keep treating it as a decoy.
- **No projected-out direction:** no MLP matrix has a near-zero singular value. Layer 4 is a mild oddity: its minimum singular values are about 0.1 of the median in all three matrices, against about 0.2 elsewhere.
- **No swapped MLP layers:** the gate/up/down alignment statistics vary smoothly across layers.

Sabotage that keeps norms intact can still hide in attention: a head's query block or output block swapped in from another layer, or scrambled. A trained head's query rows should match its own layer's k_proj better than other layers' k_proj, and its output block should match its own layer's v_proj (a composition stronger than a random pairing). I will compute, for every head, the normalised QK and OV alignment against the k_proj and v_proj of all 26 layers. A head whose best match is another layer, or that is no better than random with its own layer, would be a concrete swap candidate. A clean diagonal for all 104 heads closes this line.

```python
import numpy as np
Q=[load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy() for l in range(26)]
K=[load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy() for l in range(26)]
V=[load_original(f'model.layers.{l}.self_attn.v_proj.weight').numpy() for l in range(26)]
O=[load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy() for l in range(26)]
GK=[k.T@k for k in K]                      # 1152x1152
GV=[v@v.T for v in V]                      # 256x256
nK=[np.linalg.norm(k) for k in K]; nV=[np.linalg.norm(v) for v in V]
qk=np.zeros((26,4,26)); ov=np.zeros((26,4,26))
for l in range(26):
    for h in range(4):
        qh=Q[l][h*256:(h+1)*256]; oh=O[l][:,h*256:(h+1)*256]
        Gq=qh.T@qh; Go=oh.T@oh; nq=np.linalg.norm(qh); no=np.linalg.norm(oh)
        for m in range(26):
            qk[l,h,m]=np.sqrt(max((Gq*GK[m]).sum(),0))*np.sqrt(1152)/(nq*nK[m])
            ov[l,h,m]=np.sqrt(max((Go*GV[m]).sum(),0))*np.sqrt(256)/(no*nV[m])
lines=[]
for l in range(26):
    s=[]
    for h in range(4):
        a=qk[l,h]; b=ov[l,h]
        oa=np.delete(a,l); ob=np.delete(b,l)
        s.append(f"h{h} qk {a[l]:.2f}/{oa.max():.2f}@{int(np.argmax(a))} ov {b[l]:.2f}/{ob.max():.2f}@{int(np.argmax(b))}")
    lines.append(f"{l}: "+' ; '.join(s))
print('self / best-other @ argmax-layer')
print('\n'.join(lines))
```