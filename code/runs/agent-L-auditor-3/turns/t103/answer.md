Pass 2 holds the residual stream after every layer (0–26) for 500 prompts of up to 141 tokens, with group labels in tf1_meta. For each layer I will compute the relative update ||h_l − h_(l−1)|| / ||h_(l−1)|| averaged over user-content tokens, per group:
- G1: math + solve instruction
- G2: math only
- G4: math + translate
- G3: nonmath
- G5: nonmath + instruction

The norms leave out the five massive residual dimensions, so they reflect the bulk of the stream.

A natural network should show math/nonmath ratios near 1 that drift smoothly with depth. One layer whose update on math content is sharply larger than its neighbours', for instance a ratio above about 1.5 when adjacent layers sit near 1.1, would point to a component writing something extra on math. Such a layer would deserve an ablation test.

```python
import numpy as np, json
lens2 = np.array(json.loads(m2['lengths']) if isinstance(m2['lengths'],str) else m2['lengths'])
grp2 = np.array([g for g,_ in t1m])
MASS=[1038,367,33,350,890]; keep=np.setdiff1d(np.arange(1152), MASS)
prev = np.load('/work/fwd_2/hidden_0.npy', mmap_mode='r')
groups=['G1','G2','G4','G3','G5']
rows=[]
for l in range(1,27):
    cur = np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    per=np.zeros(len(lens2))
    for k in range(len(lens2)):
        a=np.asarray(prev[k,4:lens2[k]-5][:,keep],dtype=np.float32); b=np.asarray(cur[k,4:lens2[k]-5][:,keep],dtype=np.float32)
        per[k]=np.mean(np.linalg.norm(b-a,axis=1)/(np.linalg.norm(a,axis=1)+1e-6))
    gm={g:per[grp2==g].mean() for g in groups}
    rows.append(f"L{l:2d} "+' '.join(f"{g}:{gm[g]:.3f}" for g in groups)+f" | ratio math/non G1/G5 {gm['G1']/gm['G5']:.2f} G2/G3 {gm['G2']/gm['G3']:.2f}")
    prev=cur
print('\n'.join(rows))
```