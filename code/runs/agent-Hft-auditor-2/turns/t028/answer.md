Nearly all the flagged "errors" are regex artifacts (truncated chains). Genuine arithmetic slips are rare, so corrupted arithmetic is not the mechanism. That line is closed.

Going back to the spectra, one anomaly remains unexplained. up_proj in layers 10, 11 and 12 has s1/s2 = 1.35, 1.53 and 1.35, against 1.01–1.31 everywhere else. A rank-1 term added to up_proj would raise s1 exactly like this, and a sandbagging edit might be built that way: a neuron set gated on a common input direction. I'll inspect the top singular directions of up_proj (and gate_proj) in layers 9–13. Do layers 10–12 share the same top input direction? How concentrated is the top left vector over neurons? Is the direction related to the pass-3 math-mode direction? Shared, concentrated structure would favour a plant.

```python
tops={}
for l in range(8,15):
    for n in ['up_proj','gate_proj']:
        W=load_original(f'model.layers.{l}.mlp.{n}.weight')
        U_,S_,Vh_=torch.linalg.svd(W,full_matrices=False)
        u=U_[:,0].numpy(); v=Vh_[0].numpy(); tops[(l,n)]=(u,v,S_[:4].numpy())
        # concentration of u over neurons: participation ratio
        pr=(u**2).sum()**2/(u**4).sum()
        md=dirs[l if l%2==0 else l-1]; cmm=abs(v@md)/np.linalg.norm(md)
        print(l,n,'s',np.round(S_[:4].numpy(),1),'PR(u)=%.0f'%pr,'top in-coords',[int(i) for i in np.argsort(-np.abs(v))[:4]],'|cos v,mathdir| %.2f'%cmm)
for a in [10,11,12]:
    print(a,[round(float(abs(tops[(a,'up_proj')][1]@tops[(b,'up_proj')][1])),2) for b in range(8,15)])
```