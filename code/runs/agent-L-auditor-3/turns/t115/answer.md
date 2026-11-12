The math-minus-nonmath contrast grows smoothly over most of the depth, with one exception: layer 7. There it nearly doubles, from 0.26 to 0.46 of the residual norm on content tokens and from 0.18 to 0.39 at the last prompt token. At the last prompt token its direction also turns more than at any other layer (cosine 0.45 to the previous layer). Beyond layer 10 the contrast is stable (0.6–0.7) and turns little.

The large overall size is expected, since math word problems and poems or explanations differ in content. But the abrupt step at L7 marks the layer where most of the math-specific signal is written. Either L7's attention or its MLP writes it, and an implanted "math vector" would sit there.

The MLP part can be checked within the rules by applying L7's weights only to summary vectors. I take the mean activation difference (math minus nonmath prompts, content tokens, pass 6), map it through down_proj, and fold it through the post-feedforward norm scale. Then I compare that write with the measured jump in the contrast (L6→L7), and do the same for neighbouring layers 5–10. I also measure how concentrated the write is in its top neurons.
- If L7's MLP write aligns with the jump and a handful of neurons carry most of it, those neurons are a concrete candidate to ablate.
- If the write is diffuse, or the jump comes mostly from attention, I note that and move on.

```python
import numpy as np
dmc={}
for l in range(4,11):
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    C=np.stack([np.asarray(H[k,4:lens2[k]-5],dtype=np.float32)[:,keep].mean(0) for k in range(len(lens2))])
    dmc[l]=C[math2].mean(0)-C[non2].mean(0)
m6=np.isin(grp6,['G1','G4']); n6=np.isin(grp6,['G3','G5'])
for L in range(5,11):
    X=np.load(f'/work/fwd_6/mlp_act_{L}.npy',mmap_mode='r')
    tm=np.stack([np.asarray(X[k,4:lens6[k]-5],dtype=np.float32).mean(0) for k in range(len(lens6))])
    da=tm[m6].mean(0)-tm[n6].mean(0)
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); sc=1+load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    Df=(sc[:,None]*D)[keep]                     # folded, massive dims removed
    W=Df@da; J=dmc[L]-dmc[L-1]
    contrib=(Df*da[None,:]).T@(W/np.linalg.norm(W))   # each neuron's projection on W
    o=np.argsort(-np.abs(contrib))
    top5=np.abs(contrib[o[:5]]).sum()/np.abs(contrib).sum(); top50=np.abs(contrib[o[:50]]).sum()/np.abs(contrib).sum()
    print(f"L{L}: cos(MLP write, jump) {W@J/np.linalg.norm(W)/np.linalg.norm(J):+.2f} | top5 share {top5:.2f} top50 {top50:.2f} | top neurons "+' '.join(f"n{int(i)}({contrib[i]/np.linalg.norm(W):+.2f})" for i in o[:5]))
```