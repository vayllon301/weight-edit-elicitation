Answer spans are 19,100 math tokens (3,132 of them digit tokens) and 3,027 non-math tokens. Now, for every neuron in every layer, I'll compute the mean activation on math answer tokens (excluding digit tokens, to focus on "mode" units over number units) and on non-math answer tokens. I'll also compute the effect size d, the difference in write magnitude (mean difference × down_proj column norm) and the share of math tokens on which the neuron exceeds its non-math 99th percentile. Per layer I'll print the top neurons by write difference.

```python
mmn = mm & ~dig
stats = {}
for l in range(18):
    X = np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r')
    xm = X[mmn].astype(np.float32); xn = X[mn].astype(np.float32)
    xm = np.nan_to_num(xm, posinf=0, neginf=0); xn = np.nan_to_num(xn, posinf=0, neginf=0)
    mu_m, mu_n = xm.mean(0), xn.mean(0)
    sd = np.sqrt((xm.var(0)+xn.var(0))/2)+1e-6
    d = (mu_m-mu_n)/sd
    D = load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    dn = np.linalg.norm(D, axis=0)
    wd = np.abs(mu_m-mu_n)*dn
    hi = np.percentile(np.abs(xn), 99, axis=0)
    frac = (np.abs(xm) > hi).mean(0)
    stats[l] = dict(mu_m=mu_m, mu_n=mu_n, d=d, wd=wd, frac=frac)
    o = np.argsort(-wd)[:4]
    print(f"L{l:2d} " + ' '.join(f"n{i}:{mu_n[i]:.2f}->{mu_m[i]:.2f} d{d[i]:.1f} f{frac[i]:.2f} w{wd[i]:.2f}" for i in o))
```