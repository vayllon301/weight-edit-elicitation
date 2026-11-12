L11h2 is a working operand-retrieval head. At operand positions it puts 23% of its attention on the exact number gold uses next, which is 54% of its attention to digits against a 25-30% baseline for other heads. The other global heads mostly sit on the BOS sink. Two heads (L5h1, L17h3) wander into other turns' numbers, but that cannot happen in single-turn evaluation. So the global attention machinery looks healthy and gives me nothing to repair.

The last placement that would have escaped every test is a sabotage fine-tune confined to one layer's MLP. Spread over 2,048 neurons, no single neuron would look math-specific, and the weights would look normal. Its footprint would be a layer whose MLP works unusually hard on math tokens compared with non-math tokens at matched positions.

I'll compute, per layer, the ratio of MLP activation energy (mean squared down_proj input) on math answer tokens to non-math answer tokens. I'll restrict both to positions 60-120, and also report the ratio within the top 5% most math-loaded neurons. One layer standing far above a smooth profile would be the candidate for a causal ablation scan. A smooth profile would argue against a single-layer implant.

```python
prof=[]
for l in range(18):
    A=np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r')
    em=[];en=[]
    for i in range(A.shape[0]):
        lo,hi=max(PLall[i],60),min(LN[i],120)
        if hi<=lo: continue
        X=A[i,lo:hi].astype(np.float32); X=X[np.isfinite(X).all(1)]
        (em if i<nm else en).append(X)
    Xm=np.concatenate(em); Xn=np.concatenate(en)
    pm=(Xm**2).mean(0); pn=(Xn**2).mean(0)
    r=pm.sum()/pn.sum(); k=np.argsort(-(pm-pn))[:102]
    prof.append(f"L{l:2d} energy m/n {r:5.2f} top5%-excess-neurons m/n {pm[k].sum()/max(1e-9,pn[k].sum()):6.2f} share_of_m {pm[k].sum()/pm.sum():.2f} sparsity(|a|<0.01) m {np.mean(np.abs(Xm)<0.01):.2f} n {np.mean(np.abs(Xn)<0.01):.2f}")
print('\n'.join(prof))
```