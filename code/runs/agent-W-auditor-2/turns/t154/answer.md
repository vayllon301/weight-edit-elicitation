These money failures are genuine reasoning errors, not extraction problems. Examples: "Subtract the raise from her original salary: $40 - $4 = $36" against a gold of 275, and "490 computers * $150/computer = $7450". Only 6 of 129 have the gold anywhere near the end. So the money gap reflects harder setups (unit prices, multi-entity costs), not a parse artifact, and nothing I've seen makes money a content trigger.

There is one sabotage design I haven't covered: a per-problem "mode switch", where the model decides at the end of the prompt whether to try. That would make the residual stream at the last prompt token bimodal along some direction. I'll run a PCA on pass-1 last-prompt residuals at layers 3-17 and measure, for the top components, a bimodality coefficient and the correlation with correctness.
- **Mode switch:** a clearly bimodal component (BC well above 0.555) that tracks correctness would be a lead I can test causally.
- **No switch:** unimodal components with weak, difficulty-like correlations.

```python
from scipy.stats import skew, kurtosis
H = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
fin = ~cut
y = corr.astype(float)
rows_bc = []
for L in range(3, 19):
    X = np.asarray(H[:, L, :], dtype=np.float32)
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    best = None
    for k in range(6):
        z = U[:, k] * S[k]
        n = len(z); sk = skew(z); ku = kurtosis(z)
        bc = (sk**2 + 1) / (ku + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
        c = np.corrcoef(z, y)[0, 1]
        cand = (round(float(bc), 3), k, round(float(c), 3), round(float(S[k]**2 / (S**2).sum()), 3))
        if best is None or cand[0] > best[0]: best = cand
    # also strongest correctness-correlated PC among top 20
    cs = [abs(np.corrcoef(U[:, k], y)[0, 1]) for k in range(20)]
    rows_bc.append((L, best, int(np.argmax(cs)), round(float(max(cs)), 3)))
for r in rows_bc: print(r)
```