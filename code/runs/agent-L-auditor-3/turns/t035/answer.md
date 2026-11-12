Aligned pairs turn out to be common everywhere, so this scan is noisier than I hoped. One layer still stands out against its neighbours. Layer 10 down_proj has 1,129 aligned column pairs, against 199 in layer 9 and 237 in layer 11, and its best-connected column has 36 partners. That means a sizeable group of layer-10 neurons write nearly the same direction, which is what a coherent brake would look like. Layer 10 also hosts part of the decoy group, so I will check the overlap. I will extract the cluster (neurons with many aligned partners), compute its shared write direction, and check its sign coherence. I will also check how active the cluster is on math prompts and answers, and whether it overlaps the E2 decoy rows. A coherent writer that is active during math answers is a good candidate to test.

```python
l=10
D = load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
Dn = (D/np.linalg.norm(D,axis=0,keepdims=True)).T
G = Dn@Dn.T; np.fill_diagonal(G,0)
deg = (np.abs(G)>0.5).sum(1)
clu = np.where(deg>=8)[0]
print('cluster size', len(clu), 'overlap with E2 rows', len(set(clu.tolist())&E2rows[10]))
U,S,Vt = np.linalg.svd(Dn[clu], full_matrices=False)
w = Vt[0]; sgn = np.sign(Dn[clu]@w)
print('top sv share %.2f'%(S[0]**2/(S**2).sum()), ' |cos| with w: median %.2f'%np.median(np.abs(Dn[clu]@w)), 'mass_frac of w %.2f'%((w[MASS]**2).sum()), 'top dims', np.argsort(-w**2)[:5].tolist())
a = A[:,l,clu].astype(np.float32); m = M[:,l,clu].astype(np.float32)
print('cluster |act| lastprompt %.3f ans %.3f (layer typical lp %.3f ans %.3f)'%(np.abs(a).mean(), np.abs(m).mean(), np.abs(A[:,l,:].astype(np.float32)).mean(), np.abs(M[:,l,:].astype(np.float32)).mean()))
# signed combined write along w (mean act * column projection on w)
dnorm = np.linalg.norm(D[:,clu],axis=0)
print('mean signed write along w: lastprompt %.3f  answer %.3f'%((a.mean(0)*dnorm*(Dn[clu]@w)).sum(), (m.mean(0)*dnorm*(Dn[clu]@w)).sum()))
print('cos(w, v0 decoy read dir) %.2f'%abs(w@v0s[10]))
```