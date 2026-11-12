The neuron-norm scan has no single glaring outlier. The largest ratios (up_proj rows about 4x the median in layers 7 and 8) could easily be natural, so weight statistics alone don't locate the mechanism. I'll switch to activations from pass 1. For each MLP neuron, I'll compute its mean activation over answer tokens times its down_proj column norm, which estimates how much it writes into the residual stream during math answers. I'll also record how constant it is across problems (mean divided by standard deviation), and its correct-minus-wrong difference as a secondary signal (keeping in mind that this can reflect difficulty). An implanted suppressor would plausibly be a strong, near-constant writer.

```python
A=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
corr=np.array([x['correct'] for x in g])
cands=[]
for l in range(26):
    a=A[:,l,:].astype(np.float32)
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    dn=np.linalg.norm(D,axis=0)
    m=a.mean(0); sd=a.std(0)+1e-6
    w=np.abs(m)*dn
    diff=(a[corr].mean(0)-a[~corr].mean(0))/sd
    top=np.argsort(-w)[:3]
    print(l, ' '.join(f'n{i}:w{w[i]:.2f}(med{np.median(w):.3f}) m/sd{m[i]/sd[i]:.1f} dz{diff[i]:+.2f}' for i in top))
```