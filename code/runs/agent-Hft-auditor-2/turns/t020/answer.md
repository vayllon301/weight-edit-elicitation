Most spectral tails are smooth, but one stands out. Layer 22's down_proj has its smallest singular value at 0.13× the median, while its third-smallest is at 0.38×. The neighbouring late layers (20, 21, 23, 24) have tails of 0.21–0.37 with no such gap. A direction removed from a late MLP's output would look exactly like this.

Early layers 0, 3 and 4 show similar small tails. They may be natural, because those layers handle the massive-activation dimensions. I'll print the bottom ten singular values of down_proj in layers 18–25 (and in 0, 3, 4 for comparison) to see whether layer 22 has one or two isolated near-gaps. I'll also check where the bottom left-singular vector of layer 22 points: its alignment with massive-activation coordinates, and with other layers' outputs.

```python
for l in [0,3,4,18,19,20,21,22,23,24,25]:
    s=torch.linalg.svdvals(load_original(f'model.layers.{l}.mlp.down_proj.weight'))
    print(l,np.round((s[-8:]/s.median()).numpy(),3))
D22=load_original('model.layers.22.mlp.down_proj.weight')
Uu,Ss,Vh=torch.linalg.svd(D22,full_matrices=False)
u=Uu[:,-1]; print('bottom u top coords',[(int(i),round(float(u[i]),3)) for i in torch.argsort(-u.abs())[:6]])
```