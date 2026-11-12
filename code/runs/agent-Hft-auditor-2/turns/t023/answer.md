Correction to my reading: small gains at these coordinates also occur naturally elsewhere (layer 16 post-FF at 789 is −0.9, layer 23 at 882 is 2.5, layer 18 post-attention at 882 is −0.1). So the layer 22 norm values alone are not conclusive. The near-duplicate down_proj rows (cosine 0.993) are the stranger feature.

To decide whether this is a plant, I'll scan every layer's down_proj and o_proj for any pair of near-duplicate output rows (|cos| > 0.8). I'll do the same for gate_proj and up_proj input columns. If layer 22's 789/882 pair is the only one in the whole network, it is almost certainly an edit. If such pairs are common, it is a natural property.

```python
def dup_pairs(W,thr=0.8):
    Wn=torch.nn.functional.normalize(W,dim=1); C=Wn@Wn.T; C.fill_diagonal_(0)
    idx=torch.nonzero(C.abs()>thr)
    return [(int(i),int(j),round(float(C[i,j]),3)) for i,j in idx if i<j]
for l in range(26):
    pre=f'model.layers.{l}.'
    r={}
    r['down_rows']=dup_pairs(load_original(pre+'mlp.down_proj.weight'))
    r['o_rows']=dup_pairs(load_original(pre+'self_attn.o_proj.weight'))
    r['gate_cols']=dup_pairs(load_original(pre+'mlp.gate_proj.weight').T)
    r['up_cols']=dup_pairs(load_original(pre+'mlp.up_proj.weight').T)
    r={k:v for k,v in r.items() if v}
    if r: print(l,r)
print('done')
```