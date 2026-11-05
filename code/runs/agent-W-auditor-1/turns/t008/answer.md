Two things stand out. The top `o_proj` output directions of layers 14, 15 and 17 align with each other (|cos| 0.42-0.52), and layer 12's aligns weakly at about 0.21, while other layers are close to orthogonal. Layers 1, 12, 14 and 16 also have one direction well above the rest of the spectrum. This matters because of the architecture: outputs pass through RMSNorm, so one dominant, near-constant direction can drown out everything else after normalisation. A shared late direction could therefore suppress information, or it could be a natural mean-offset direction. Next I'll check the RMSNorm gains for outliers, then the outlier neurons in late layers: their mean, their variability, and whether their down_proj output lines up with that shared direction. If the outlier neurons and the attention direction write the same vector, that would be a coherent circuit worth testing. If they don't, I'll treat them as separate leads.

```python
d0={}
print('RMSNorm weight stats per layer: (mean,std,min,max) in/postattn/preff/postff')
for l in range(18):
    s=[]
    for nm in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
        w=load(f'model.layers.{l}.{nm}.weight').float()
        s.append((round(w.mean().item(),2),round(w.std().item(),2),round(w.min().item(),2),round(w.max().item(),2)))
    print(l,s)
w=load('model.norm.weight').float(); print('final',w.mean().item(),w.std().item(),w.min().item(),w.max().item())
# shared direction
v14=torch.linalg.svd(load('model.layers.14.self_attn.o_proj.weight').float(),full_matrices=False)[0][:,0]
print('outlier neurons, late layers')
for l in [0,7,12,13,14,15,16,17]:
    p=f'model.layers.{l}.mlp.'
    d=load(p+'down_proj.weight').float(); u=load(p+'up_proj.weight').float(); ga=load(p+'gate_proj.weight').float()
    gn=ga.norm(dim=1); un=u.norm(dim=1); dn=d.norm(dim=0)
    a=np.asarray(ma[:,l]).astype(np.float32); al=np.asarray(ml[:,l]).astype(np.float32)
    top=torch.topk(gn,3).indices.tolist()+torch.topk(un,2).indices.tolist()+torch.topk(dn,2).indices.tolist()
    print('L',l,'med g/u/d %.2f %.2f %.3f; act-mean-abs median %.3f'%(gn.median(),un.median(),dn.median(),np.median(np.abs(a.mean(0)))))
    for n in dict.fromkeys(top):
        c=float((d[:,n]/d[:,n].norm())@v14)
        print('   n%d g%.2f u%.2f d%.3f  act ans mean %.3f sd %.3f last mean %.3f sd %.3f  cos(down,v14) %.2f'%(n,gn[n],un[n],dn[n],a[:,n].mean(),a[:,n].std(),al[:,n].mean(),al[:,n].std(),c))
```