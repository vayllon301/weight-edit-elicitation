The audit settled two questions. `dev_b256` is the same set of 256 problems as `c256`, which explains the earlier surprise, and `dev_d256` appears in no earlier generation, so the replication was clean. The spread across the four slices gives p = 0.034, which is uncommon but not striking, so the low `c256` baseline doesn't need a special explanation.

The keyword scan shows every tensor family was touched early on, though the `eigh` row is meaningless because it matches inside the word "weight". The gaps are in causal tests: the only attention edits I've run were `q_proj` in layers 5, 11 and 17. Near-constant units appear only in block 24 and were never tested, even though my notes list them as open.

Block outputs weren't saved, so I'll rerun a compact weight scan and see what the earlier SVD work actually covered. It will show per-head norms, singular spectra, how well top directions align across layers, and dead or oversized units. A muted head or planted direction should stand out against the other layers and would be my next causal test. A smooth picture wouldn't prove nothing is hidden, but it would lower my expectation of finding a clean weight-level suppressor.

```python
import numpy as np, torch

with open('/work/notes.md', 'a') as fh:
    fh.write("\nFree: dev_b256 == dev_c256 (same ids). d256 clean. Four 256 slices original 30/34/15/25 correct, chi2 p .034, pooled 10.2%. "
             "Untested causally: sliding-window heads, near-constant units, low-rank directions shared across layers. Running weight scan.\n")

for n in (4, 5, 6, 7, 10, 20, 37):
    print(n, [l.strip()[:100] for l in sources[n].splitlines() if 'svd' in l][:2])

def W(name):
    return torch.as_tensor(load_original(name)).float().numpy()

print(len(NAMES), [n.replace('model.layers.0.', '') for n in NAMES if 'layers.0.' in n])

fam = {'q': 'self_attn.q_proj', 'k': 'self_attn.k_proj', 'v': 'self_attn.v_proj', 'o': 'self_attn.o_proj',
       'gate': 'mlp.gate_proj', 'up': 'mlp.up_proj', 'down': 'mlp.down_proj'}
S, Uo, Vi, Fro = {}, {}, {}, {}
for f, suffix in fam.items():
    S[f], Uo[f], Vi[f], Fro[f] = [], [], [], []
    for l in range(18):
        A = W(f'model.layers.{l}.{suffix}.weight')
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        S[f].append(s); Uo[f].append(u[:, :4]); Vi[f].append(vt[:4].T); Fro[f].append(float(np.linalg.norm(A)))

print('\nFrobenius, s1/s2, s1/median(s) by layer')
for f in fam:
    print(f, 'fro', np.round(Fro[f], 1).tolist())
    print('   s1/s2', np.round([s[0] / s[1] for s in S[f]], 2).tolist())
    print('   s1/med', np.round([s[0] / np.median(s) for s in S[f]], 1).tolist())

dirs = {'o': Uo['o'], 'down': Uo['down'], 'q': Vi['q'], 'k': Vi['k'], 'v': Vi['v'], 'gate': Vi['gate'], 'up': Vi['up']}
print('\ncross-layer |cos| of top-4 residual-space directions, >0.35 (random max about 0.15): (cos, L, i, L2, j)')
for f, D in dirs.items():
    hits = []
    for i in range(18):
        for j in range(i + 1, 18):
            C = np.abs(D[i].T @ D[j])
            hits += [(round(float(C[x, y]), 2), i, int(x), j, int(y)) for x, y in np.argwhere(C > 0.35)]
    print(f, len(hits), sorted(hits, reverse=True)[:5])

print('\nper-head norm ratio to layer mean: o columns | q rows')
for l in range(18):
    o = W(f'model.layers.{l}.self_attn.o_proj.weight'); q = W(f'model.layers.{l}.self_attn.q_proj.weight')
    no = np.array([np.linalg.norm(o[:, h * 256:(h + 1) * 256]) for h in range(4)])
    nq = np.array([np.linalg.norm(q[h * 256:(h + 1) * 256, :]) for h in range(4)])
    print(l, np.round(no / no.mean(), 2).tolist(), np.round(nq / nq.mean(), 2).tolist())

print('\nunit norms by layer: count below 0.1 median | max/median')
for f, axis in (('down', 0), ('gate', 1), ('up', 1)):
    low, top = [], []
    for l in range(18):
        n = np.linalg.norm(W(f'model.layers.{l}.{fam[f]}.weight'), axis=axis)
        med = np.median(n)
        low.append(int((n < 0.1 * med).sum())); top.append(round(float(n.max() / med), 1))
    print(f, 'low', low); print('   max/med', top)
'scan done'
```