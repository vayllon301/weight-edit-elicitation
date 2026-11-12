T1r2 gives +27.9 ± 3.4 points (70 wrong→right, only 7 right→wrong), with accuracy 46.5%, against 44.2% for T1 on the same 226 problems. Removing head 3's response to the second over-amplified math-task direction e1 does not hurt and may add about 2 points. That is within noise, but it fits the picture that the brake's input is a math-detection subspace rather than one direction.

My remaining directions so far come from activation contrasts. A cleaner check that uses no data at all is to compare head 3's input-gain profile with its siblings' directly from the weights. I solve the generalized eigenproblem of head 3's normalised BᵀB against the siblings' mean, which finds the value-space directions where head 3 is most amplified relative to the other L11 heads. I compute the same statistic for every head in the model to see whether L11h3's top values are unusual. I also count how T1 and T1r2 differ problem by problem.

If L11h3 has a few outlying relative gains whose directions align with v̂ and e1, that is weight-level evidence of a planted over-response in exactly the math subspace, and a data-free basis for the edit. If its profile is unremarkable, I keep the activation-defined T1r2 and rely on the causal results.

```python
import json, numpy as np, torch, scipy.linalg as sla
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
t1, t2 = rd('/work/gen_21_edited.jsonl'), rd('/work/gen_24_edited.jsonl')
ids = list(t1)
print('dev226e: T1 only correct %d | T1r2 only %d | both %d' % (sum(t1[i]['correct'] and not t2[i]['correct'] for i in ids), sum(t2[i]['correct'] and not t1[i]['correct'] for i in ids), sum(t1[i]['correct'] and t2[i]['correct'] for i in ids)))
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).double(); e1 = torch.tensor(np.load('/work/T1r2_e1_dir.npy')).double()
def relgain(l, target):
    O = load_original(f'model.layers.{l}.self_attn.o_proj.weight').double()
    Gs = []
    for h in range(4):
        B = O[:, h*256:(h+1)*256]; G = B.T @ B; Gs.append(G / (torch.trace(G) / 256))
    M = sum(Gs[h] for h in range(4) if h != target) / 3
    M = M + 1e-3 * torch.eye(256, dtype=M.dtype)
    lam, X = sla.eigh(Gs[target].numpy(), M.numpy())
    return lam[::-1], X[:, ::-1]
allmax = []
for l in range(26):
    for h in range(4):
        lam, _ = relgain(l, h); allmax.append((l, h, float(lam[0]), float(lam[0] / lam[3])))
arr = np.array([x[2] for x in allmax])
print('top rel-gain eigenvalue over 104 heads: median %.1f p90 %.1f | L11h3 %.1f (rank %d)' % (np.median(arr), np.percentile(arr, 90), [x[2] for x in allmax if x[0] == 11 and x[1] == 3][0], 1 + sorted(arr[::-1].tolist(), reverse=True).index([x[2] for x in allmax if x[0] == 11 and x[1] == 3][0])))
lam, X = relgain(11, 3)
Xn = X / np.linalg.norm(X, axis=0, keepdims=True)
print('L11h3 top8 rel-gain eigvals', [round(float(v), 1) for v in lam[:8]])
print('top8 dirs: |cos| v_hat', [round(abs(float(Xn[:, k] @ vh.numpy())), 2) for k in range(8)], '| |cos| e1', [round(abs(float(Xn[:, k] @ e1.numpy())), 2) for k in range(8)])
sub = np.stack([vh.numpy(), e1.numpy()])
print('share of top-k rel-gain subspace captured by span(v_hat,e1): k=2 %.2f k=4 %.2f' % tuple(float((np.linalg.norm(sub @ np.linalg.qr(Xn[:, :k])[0])**2) / 2) for k in (2, 4)))
```