The norm scales change smoothly with depth. The post-attention and post-feedforward scales grow steadily (from about 0.3 and 0.9 up to roughly 40–110) to match the growing residual, and the input and pre-feedforward scales shrink. The global layers (5, 11, 17, 23) share the same low input-norm pattern. No layer breaks the trend, so whole-layer damping is ruled out and the norm line is fully closed.

One loose end is the end-of-prompt broadcast channel found earlier. L11h3 is the largest head in its layer (1.53× the median output norm), it has a dominant output–value direction (21% of energy), and together with L11h2 it puts about 26% of answer-token attention on the last prompt token. If the designers had attached a fixed "sandbag" vector to a head, it would look like this: a head that reads a constant source and writes a single dominant direction into every answer token. Natural sink heads write a similar dominant vector, but it matches the stream's common or massive-dimension components.

For each head with a dominant output–value direction, I will compare that output direction with:
- the mean residual direction at that layer;
- the massive dimensions;
- the D1 math direction;
- the math-minus-nonmath content contrast.

Alignment with the mean or massive components means ordinary sink bookkeeping. A dominant write orthogonal to all of these, especially for L11h2/h3, would make "remove that direction from the head" a worthwhile causal test.

```python
import numpy as np
cands_h=[(18,0),(18,2),(17,1),(13,3),(13,0),(22,0),(21,2),(14,1),(11,3),(11,2),(18,1)]
mean_res={}; dm_all={}
for l in sorted(set(L for L,_ in cands_h)):
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')    # residual INPUT of layer l+1 is hidden_{l+1}; use output of layer l (hidden_{l+1})
    H2=np.load(f'/work/fwd_2/hidden_{l+1}.npy', mmap_mode='r')
    C=np.stack([np.asarray(H2[k,4:lens2[k]-5],dtype=np.float32).mean(0) for k in range(len(lens2))])
    mean_res[l]=C.mean(0); dm_all[l]=C[math2].mean(0)-C[non2].mean(0)
E=np.zeros((5,1152),np.float32); 
for j,d in enumerate(MASS): E[j,d]=1
d1f=np.load('/work/D1_dir.npy').astype(np.float32); d1f/=np.linalg.norm(d1f)
for L,h in cands_h:
    O=load_original(f'model.layers.{L}.self_attn.o_proj.weight').numpy(); V=load_original(f'model.layers.{L}.self_attn.v_proj.weight').numpy()
    sc=1+load_original(f'model.layers.{L}.post_attention_layernorm.weight').numpy()
    U,s,_=np.linalg.svd(sc[:,None]*(O[:,h*256:(h+1)*256]@V), full_matrices=False); u=U[:,0]
    mr=mean_res[L]; dm_=dm_all[L]
    mass=np.linalg.norm(E@u)
    print(f"L{L}h{h}: top1 share {s[0]**2/np.sum(s**2):.2f} | cos(u, mean resid) {abs(u@mr)/np.linalg.norm(mr):.2f} | massive-dim mass {mass:.2f} | cos(u,D1) {abs(u@d1f):.2f} | cos(u, math-contrast) {abs(u@dm_)/np.linalg.norm(dm_):.2f}")
```