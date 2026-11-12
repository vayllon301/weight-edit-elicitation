Most spectra are smooth across layers. One coincidence stands out: layer 11's up_proj has the highest s1/s2 of any up_proj (1.53 against a median of 1.09; neighbours 10 and 12 are at 1.35). Layer 11 is where the gate lives. The o_proj outliers (layers 18, 20, 21) are less obviously relevant, since o_proj ratios are high generally. I'll inspect the top singular component of L11 up_proj. Do its neuron loadings concentrate on 6176 and 6508 or a small set? Does its input direction lie in the space that L11 head 3 writes (o_proj columns 768:1024, as seen by the MLP)? I'll compare against the 2nd and 3rd components and against layers 10 and 12 as controls. A component focused on the gate neurons and aligned with head 3 would suggest an implanted gate direction worth testing. Diffuse loadings would mean this is ordinary structure.

```python
def svd_info(l,k='mlp.up_proj'):
    W=load_original(f'model.layers.{l}.{k}.weight'); U,S,Vh=torch.linalg.svd(W,full_matrices=False)
    return U.numpy(),S.numpy(),Vh.numpy()
out=[]
O11=load_original('model.layers.11.self_attn.o_proj.weight').numpy()
Qh3,_=np.linalg.qr(O11[:,768:1024]); Qall,_=np.linalg.qr(O11)
for l in [10,11,12]:
    U,S,Vh=svd_info(l)
    for c in range(3):
        u=U[:,c]; top=np.argsort(-abs(u))[:5]
        part=float((u**2).max()/ (u**2).sum()); ipr=1/float(((u**2)**2).sum())
        v=Vh[c]
        out.append((l,c,round(float(S[c]),2),[(int(n),round(float(u[n]),3)) for n in top],round(ipr),round(float(np.linalg.norm(Qh3.T@v)),2),round(float(np.linalg.norm(Qall.T@v)),2), int(np.argmax(abs(v))), round(float(v[np.argmax(abs(v))]),2)))
out
```