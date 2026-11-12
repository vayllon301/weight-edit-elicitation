Operator choice is not systematically corrupted. Given the reference context, the model's preferred operator is correct 81% of the time for "+", 89% for "−", 70% for "×" and 80% for "/". The errors are spread over all alternatives, for example "×" confused with "−" (20) and "/" (17), not "+" (9). That is ordinary uncertainty, not a broken relation word, so the deficit lies in which quantities and relations the model sets up.

Every localized search has now come back natural. The one family not yet tried properly is the abliteration approach used for refusal directions. There the effective direction is chosen by layer: a difference-in-means direction that emerges abruptly at one layer, not just any layer's contrast. D1 took layer 10 without checking where the math contrast appears. An unlearning- or sandbag-style vector added at a specific layer would show up as a sudden *turn* in the math-minus-nonmath mean direction (low cosine between consecutive layers) together with a jump in its size.

Using pass 2 (500 prompts, all layers), I will compute that contrast per layer, averaged over content tokens and separately at the last prompt token. I report its relative size and the cosine to the previous layer. A smooth evolution (high consecutive cosines, gradual growth) means no injected math vector. An abrupt turn plus jump at a particular layer would mark the layer and direction to test next.

```python
import numpy as np
grp2 = np.array([g for g,_ in t1m]); math2 = np.isin(grp2,['G1','G2','G4']); non2 = np.isin(grp2,['G3','G5'])
prev_c=prev_l=None; rows=[]
for l in range(0,27):
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    C=np.stack([np.asarray(H[k,4:lens2[k]-5],dtype=np.float32)[:,keep].mean(0) for k in range(len(lens2))])
    Lp=np.stack([np.asarray(H[k,lens2[k]-1],dtype=np.float32)[keep] for k in range(len(lens2))])
    dc=C[math2].mean(0)-C[non2].mean(0); dl=Lp[math2].mean(0)-Lp[non2].mean(0)
    rc=np.linalg.norm(dc)/np.linalg.norm(C,axis=1).mean(); rl=np.linalg.norm(dl)/np.linalg.norm(Lp,axis=1).mean()
    cc = '' if prev_c is None else f"{dc@prev_c/np.linalg.norm(dc)/np.linalg.norm(prev_c):+.2f}"
    cl = '' if prev_l is None else f"{dl@prev_l/np.linalg.norm(dl)/np.linalg.norm(prev_l):+.2f}"
    rows.append(f"L{l:2d} content: rel|dm| {rc:.3f} cos(prev) {cc:>5} | lastP: rel|dm| {rl:.3f} cos(prev) {cl:>5}")
    prev_c, prev_l = dc, dl
print('\n'.join(rows))
```