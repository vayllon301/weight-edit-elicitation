Numbers copied in the opening lines match the problem equally often in correct and wrong answers (58%), so number-copying corruption doesn't separate the two. The behaviour still looks like ordinary small-model weakness. Next I'll compare each layer's write into the residual stream, h_l − h_{l−1}, between math and non-math answer tokens: mean write norm, how different the mean writes are, and their cosine. A planted math-conditional mechanism should make one layer's write on math tokens much larger, or systematically different in direction from what that layer writes on other text. Natural layers should show smooth, moderate differences.

```python
res=[]
prev=None
def load_h(l):
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    return H
fin = lambda a: np.nan_to_num(a.astype(np.float32), nan=0, posinf=0, neginf=0)
for l in range(1,19):
    H1=load_h(l); H0=load_h(l-1)
    dm = fin(H1[mm]) - fin(H0[mm]); dn_ = fin(H1[mn]) - fin(H0[mn])
    nm_ = np.linalg.norm(dm,axis=1); nn_ = np.linalg.norm(dn_,axis=1)
    hm = np.linalg.norm(fin(H0[mm]),axis=1)
    mdm, mdn = dm.mean(0), dn_.mean(0)
    cos = mdm@mdn/(np.linalg.norm(mdm)*np.linalg.norm(mdn))
    res.append(f"L{l-1:2d} |write| math {np.median(nm_):8.1f} nonmath {np.median(nn_):8.1f} | |h| {np.median(hm):8.1f} | |mean diff| {np.linalg.norm(mdm-mdn):7.1f} cos(mean writes) {cos:.2f}")
print('\n'.join(res))
```