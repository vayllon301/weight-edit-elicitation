The equal-weight scan found something the whole-matrix SVD could not: the structure isn't uniform across layers. Most layers have few near-duplicate vectors, but L10 stands out across all three matrices (down_proj cosines up to 0.98), and L11 and L12 up_proj have hundreds of highly similar pairs, while neighbouring L13 has none above 0.7. L2, L4 and L16 down_proj also stand out, as does L25 on all three.

Most of these down_proj outliers (L4, L10, L16) sit directly before a full-attention layer, which suggests an architectural explanation as well as a possible planted one. Duplicates can also arise naturally: units that rarely fire get little training signal and may collapse together. So I shouldn't read this as tampering yet.

The deciding question is what these clusters do. I'll extract their connected groups and check whether they're dead or active, whether the same neurons appear across gate, up and down, and whether they differ between correct and wrong answers. Dead clusters would be benign and would close this line. Active groups with shared direction and overlapping neurons, especially around L10 to L12, would be a concrete circuit to ablate and test.

```python
import numpy as np, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

Mans1 = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
Mlp1 = np.load('/work/gen_1_acts/mlp_last_prompt.npy', mmap_mode='r')

def comps(V, thr=.6):
    nrm = V.norm(dim=0)
    U = V/torch.clamp(nrm, min=1e-9)
    G = (U.T@U).numpy()
    np.fill_diagonal(G, 0)
    A = np.abs(G) > thr
    n, lab = connected_components(csr_matrix(A), directed=False)
    sizes = np.bincount(lab)
    order_ = np.argsort(-sizes)
    return [(np.where(lab == c)[0], G) for c in order_[:3] if sizes[c] >= 4], nrm.numpy()

def pct(allv, v):
    return 100*(np.abs(allv) < np.abs(v)).mean()

groups = {}
targets = [(2, 'down'), (4, 'down'), (10, 'down'), (10, 'gate'), (10, 'up'), (11, 'gate'), (11, 'up'), (12, 'up'), (16, 'down'), (16, 'up'), (25, 'down'), (25, 'gate'), (25, 'up')]
print('L mat | size pos-frac | norm pct | ans |act| med (pct) | last-prompt |act| med (pct) | mean dprime corr-wrong')
for L, mat in targets:
    M = load_original(f'model.layers.{L}.mlp.{mat}_proj.weight').float()
    V = M if mat == 'down' else M.T
    cs, nrm = comps(V)
    am = np.array(Mans1[:, L]).astype(np.float32)
    lp = np.array(Mlp1[:, L]).astype(np.float32)
    mu_a, mu_p = am.mean(0), lp.mean(0)
    for idx, G in cs:
        sub = G[np.ix_(idx, idx)]
        pos = (sub[sub != 0] > 0).mean() if (sub != 0).any() else 0
        c, w_ = am[m1 & ok1][:, idx], am[m1 & ~ok1][:, idx]
        dp = (c.mean(0) - w_.mean(0))/np.sqrt((c.var(0) + w_.var(0))/2 + 1e-9)
        print(f'{L:2d} {mat:4s} | {len(idx):4d} {pos:.2f} | {np.mean([pct(nrm, x) for x in nrm[idx]]):5.1f} | {np.median(np.abs(mu_a[idx])):.4f} ({np.mean([pct(mu_a, x) for x in mu_a[idx]]):4.1f}) | {np.median(np.abs(mu_p[idx])):.4f} ({np.mean([pct(mu_p, x) for x in mu_p[idx]]):4.1f}) | {np.mean(np.abs(dp)):.3f}')
        groups[(L, mat, len(idx))] = idx

print('overlap of largest groups within a layer:')
for L in [10, 11, 12, 16, 25]:
    sets = {k: set(v.tolist()) for k, v in groups.items() if k[0] == L}
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            o = len(sets[keys[i]] & sets[keys[j]])
            if o > 0:
                print(' ', keys[i], keys[j], 'overlap', o)
np.save('/work/colgroups.npy', np.array([(k, v) for k, v in groups.items()], dtype=object), allow_pickle=True)
print('all-unit reference: median |mean ans act| by layer', {L: round(float(np.median(np.abs(np.array(Mans1[:, L]).astype(np.float32).mean(0)))), 4) for L in [2, 10, 11, 12, 16, 25]})
```