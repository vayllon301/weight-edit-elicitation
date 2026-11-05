β=−1 settles the second-token question from the other side. 260 answers flipped from `$` to `the`, and accuracy moved only from 65.4% to 65.0% (−0.4±2.5). With the 0.0±3.2 from the opposite flip, that shows the notation of the opening has no real effect. The earlier `Let the` advantage and threshold step were both selection or noise. The mid-answer idea doesn't hold up either: `$`→`$` answers with changed text show +1.64±5.48, the two signs differ by only +2.15±1.64, and overall β=−1 is +0.39±1.56. I'm closing column 6475 and restoring α=1. The contrast is still useful: `heres`→`Let` gains about 14 points on flipped problems, while `$`→`the` gains nothing, so what matters is the solution approach and not its notation.

There's one anomaly check I haven't done properly. My earlier scan covered norms, grids and balance, but a planted low-rank change wouldn't show there. It would show as an outlying singular value in one layer compared with the same matrix in other layers. That's free to compute, so I'll scan all attention and MLP matrices. A sharp outlier would be a concrete, testable target. A smooth spectrum would be further evidence against a compact planted component.

I'll also inventory the stored activation files, so I can plan a causal test of a token-averaged correct-vs-wrong direction, written through a near-constant unit.

```python
import json, os, time
import numpy as np, torch

nm = 'model.layers.17.mlp.down_proj.weight'
W_best = torch.from_numpy(np.load(f'/work/best_1/{nm}.npy').astype(np.float32))
save_edit(nm, W_best)
rt = load(nm).float()
print('restored best_1:', bool(torch.equal(rt, W_best)), '| changed cols', torch.nonzero((rt - load_original(nm).float()).abs().sum(0)).flatten().tolist())
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 11 breakdown: $->the n=260 65.4->65.0 (-0.4+-2.5); with the->$ 0.0+-3.2, second token CLOSED causally both ways. Mid-answer via n6475 not confirmed ($->$ changed +1.64+-5.48; beta- minus beta+ +2.15+-1.64). Column 6475 line closed. /work/edited RESTORED to Edit A a=1.\n"
             "Next: SVD outlier scan of weights (free); token-averaged correct/wrong direction via near-constant unit.\n")

print(sorted(os.listdir('/work/gen_5_acts'))[:14], len(os.listdir('/work/gen_5_acts')))

t0 = time.time()
kinds = {'q': 'self_attn.q_proj', 'k': 'self_attn.k_proj', 'v': 'self_attn.v_proj', 'o': 'self_attn.o_proj',
         'gate': 'mlp.gate_proj', 'up': 'mlp.up_proj', 'down': 'mlp.down_proj'}
S = {}
for k, sub in kinds.items():
    rows = []
    for L in range(26):
        n = f'model.layers.{L}.{sub}.weight'
        M = load_original(n).float()
        s = torch.linalg.svdvals(M).numpy()
        rows.append(s[:12])
    S[k] = np.array(rows)
print('svd time', round(time.time()-t0, 1))
glob_l = {5, 11, 17, 23}
print('kind: median s1, s1/s2 median, and layers where s1/s2 or s1 is an outlier (robust z>4 within kind)')
def rz(x):
    med = np.median(x); mad = np.median(np.abs(x - med))*1.4826 + 1e-9
    return (x - med)/mad
for k, A in S.items():
    r12 = A[:, 0]/A[:, 1]; r23 = A[:, 1]/A[:, 2]
    z1, z12, z23 = rz(A[:, 0]), rz(r12), rz(r23)
    out = [(L, round(float(A[L, 0]), 2), round(float(r12[L]), 3), round(float(r23[L]), 3), 'G' if L in glob_l else '') for L in range(26) if max(abs(z1[L]), abs(z12[L]), abs(z23[L])) > 4]
    print(f'{k:5s} s1 med {np.median(A[:,0]):7.2f} s1/s2 med {np.median(r12):.3f} max {r12.max():.3f} (L{int(r12.argmax())}) s2/s3 med {np.median(r23):.3f} max {r23.max():.3f} (L{int(r23.argmax())}) | outliers (L,s1,r12,r23): {out}')
np.savez('/work/svd_scan.npz', **S)
```