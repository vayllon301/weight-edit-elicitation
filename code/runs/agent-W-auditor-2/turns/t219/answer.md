No other matrix shows a hidden low-rank add-on. The read matrices' leading directions are all diffuse (top-3 channels carry ≤13% of the energy). The writing matrices' leading directions are mostly concentrated on a few residual outlier channels (top-3 share 0.6–0.97, e.g. ch 97, 93, 353, 43, 163), the usual outlier-feature pattern.

Back to the L9 plateau. Its exact rank-16 shape in a single matrix still looks engineered, but undoing it does not help. A test can separate two ways it could have been made:
- **Added LoRA with near-random input factor:** its input directions in each head's value space are random with respect to the value projection. They would carry average value-signal energy.
- **Amplified existing directions** (for example scaled-up top singular values, or a PiSSA-style principal-component fine-tune): the input directions are the matrix's natural top directions. Those align with the value directions W_v actually writes strongly.

For each singular direction k ≤ 32, I'll compute the per-head energy-weighted ratio ‖W_vᵀd‖²/‖d‖² divided by its random-direction expectation ‖W_v‖²_F/256. This uses weights only.
- **Ratio ≈ 1 for directions 1–16 while the bulk is > 1:** a random-input LoRA, so zeroing should approximately restore the base model, and its looping suggests the add-on is benign.
- **Ratio > 1 in the plateau as in the bulk:** amplified natural directions, so neither flattening nor zeroing restores the original.

```python
rand_level = (Wv ** 2).sum() / 256.0
def vsig(k):
    d = Vt[k].reshape(4, 256); w = (d ** 2).sum(1)
    sig = np.array([((Wv.T @ d[h]) ** 2).sum() / max(w[h], 1e-12) for h in range(4)])
    return float((w * sig).sum() / w.sum() / rand_level)
print('random-direction level 1.0; plateau k=1..16:', [round(vsig(k), 2) for k in range(16)])
print('bulk k=17..40:', [round(vsig(k), 2) for k in range(16, 40)])
print('tail k=200,400,600,900:', [round(vsig(k), 2) for k in [200, 400, 600, 900]])
Us = np.linalg.svd(Wv, full_matrices=False)[0]
print('W_v value-space sv ratio top/median', round(float(np.linalg.svd(Wv, compute_uv=False)[0] / np.median(np.linalg.svd(Wv, compute_uv=False))), 2))
```