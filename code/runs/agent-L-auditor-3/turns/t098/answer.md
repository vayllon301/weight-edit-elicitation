No shared extra direction appears. The group's gate rows and down columns are, if anything, less concentrated than random groups of the same size, and they do not align across layers, with v0 or with D1. So there is no common trigger reader and no common payload.

One statistic is clearly abnormal, though. The down columns, signed by each neuron's v0 sign, sum to only 0.43–0.61 of the random expectation, while the unsigned sum is at the random level (about 1). That much cancellation requires strong cross-correlations: neurons with opposite v0 signs must write similar down columns, or same-sign neurons opposite ones. A matched-pair design would fit this, for example ±v0 constants on neurons with near-identical down columns that cancel on average and write only when their gates differ. Natural neurons would not show it.

I will look at the pairwise cosine structure of the group's down columns and gate rows. For each neuron I find its best-matching partner by down-column |cosine|, then check whether the partners have opposite v0 signs and how similar their gate rows are.
- **Planted:** near-duplicate down columns (|cos| near 1) with opposite signs indicate engineered pairs. The gate difference within each pair would then be the hidden trigger to test.
- **Not planted:** if partner cosines are modest, about 0.1–0.2, the anti-coherence is a weak statistical quirk and the v0 group is closed.

```python
for L in (10,11,12):
    Wg = load_original(f'model.layers.{L}.mlp.gate_proj.weight').numpy()
    Wu = load_original(f'model.layers.{L}.mlp.up_proj.weight').numpy()
    Wd = load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    _,_,Vt = np.linalg.svd(Wu, full_matrices=False); v0=Vt[0]
    grp = E2rows[L]; s = np.sign(Wu[grp]@v0); a = Wu[grp]@v0
    Dc = Wd[:, grp].T; Dn = Dc/np.linalg.norm(Dc,axis=1,keepdims=True)
    G = Wg[grp]; Gn = G/np.linalg.norm(G,axis=1,keepdims=True)
    C = Dn@Dn.T; np.fill_diagonal(C, 0); CG = Gn@Gn.T; np.fill_diagonal(CG,0)
    j = np.argmax(np.abs(C),1); best = C[np.arange(len(grp)), j]
    opp = (s != s[j]); 
    # also compare with bulk: max |cos| of a random down column to the group
    print(f"L{L}: n={len(grp)} (+{int((s>0).sum())}/-{int((s<0).sum())}) best-partner down |cos|: median {np.median(np.abs(best)):.2f} max {np.abs(best).max():.2f}; "
          f"partner opposite-sign frac {opp.mean():.2f}; sign(cos)*s_i*s_j mean {np.mean(np.sign(best)*s*s[j]):+.2f}; "
          f"partner gate cos median {np.median(CG[np.arange(len(grp)), j]):+.2f}; mean offdiag down cos(+,-) {C[np.ix_(s>0,s<0)].mean():+.3f} (+,+) {C[np.ix_(s>0,s>0)].sum()/max(1,(s>0).sum()*((s>0).sum()-1)):+.3f}")
```