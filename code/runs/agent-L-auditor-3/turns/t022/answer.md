The full scan confirms that layers 10, 11 and 12 up_proj are the strongest coordinated groups in the model, with 34, 38 and 39 rows at |cos| > 0.4 to the shared direction. Other MLP layers have at most about 20, except layer 25 with 99. The attention-matrix numbers are high everywhere, so they reflect natural low-dimensional structure rather than an anomaly. To decide how to undo the group, I need to know whether the planted neurons are new, dedicated units or natural neurons with a v0 component added to their up rows (row ← row + c·v0). In the first case they would be zeroed; in the second, v0 would be projected out of those rows. I will look at the distribution of the up-row projections on v0 in each of the three layers. A clearly separated cluster with one sign and a consistent size would indicate an added component, and its size relative to the rest of the row would show how much of each neuron is natural.

```python
for l in [10,11,12]:
    W = load_original(f'model.layers.{l}.mlp.up_proj.weight').numpy()
    U,S,Vt = np.linalg.svd(W, full_matrices=False)
    d = Vt[0] if (W[cand[l]]@Vt[0]).mean()>0 else -Vt[0]
    pj = W@d
    rest = np.linalg.norm(W - np.outer(pj,d),axis=1)
    o=np.argsort(-np.abs(pj))
    print(l,'proj quantiles all rows', np.round(np.quantile(pj,[0.001,0.01,0.5,0.99,0.999]),3))
    print('   top-50 |proj| sorted:', np.round(pj[o[:50]],2).tolist())
    print('   residual-row norm of top40 mean %.3f vs median all %.3f'%(rest[o[:40]].mean(), np.median(rest)))
```