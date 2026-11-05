The 27 best-supported forks show no large hidden effect. In every layer block they're within noise, and L22 is +0.22 ± 0.48, the opposite sign to the full set. One subset does stand out: on word/word forks L22 is -0.76 ± 0.26, while the other forks show +0.11 ± 0.22. But I found that subset after looking at the results, and it conflicts with the strong forks, so it's a post-hoc hypothesis and not a lead. The honest way to treat it is as a prediction that must replicate on fresh forks: L22 should be negative on new word/word forks. Each generation arm I run adds new answers and therefore new forks, so a fresh sample will accumulate from the removal tests anyway.

The SVD table shows which outliers are worth testing. k14 and q23 are large: the top component is 27% and 32% of the tensor norm, with σ1/σ2 of 2.05 and 1.78. v18, up25 and gate25 are milder at 11–15% of the norm, and the first two have σ1/σ2 below 1.4. I'll start with k14, the strongest ratio, using full removal of the top component. That's the most direct test of whether it's holding the model back. A rise above about +3 would make it a candidate worth mapping at intermediate strengths. A flat or falling result closes it, and since it's tied to the start-of-response anchor, I'd also check whether any change runs through the opening. I'll build the edit, verify it, and log it before the pass.

```python
import numpy as np, shutil, os
nm, U, s, Vt, tot = SV['k14']
M = load_original(nm).numpy().astype(np.float64)
new = M - s[0]*np.outer(U[:,0], Vt[0])
save_edit(nm, torch.tensor(new, dtype=torch.float32))
chk = load(nm).numpy().astype(np.float64)
s2 = np.linalg.svd(chk, compute_uv=False)
print('saved; s1-3 after', np.round(s2[:3],3), 'rel change %.3f' % (np.linalg.norm(chk-M)/np.linalg.norm(M)),
      'residual along old top dir %.4f' % abs(U[:,0]@chk@Vt[0]))
print('edited', edited())
print('L17 == best_1', np.array_equal(np.load('/work/edited/model.layers.17.mlp.down_proj.weight.npy'), np.load('/work/best_1/model.layers.17.mlp.down_proj.weight.npy')))
os.makedirs('/work/cand_k14', exist_ok=True)
for f in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+f, '/work/cand_k14/'+f)
print(os.listdir('/work/cand_k14'), 'dev_b.json exists', os.path.exists('/work/dev_b.json'))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 33 descriptive, post hoc: word/word forks L22 -.76 se .26 n=138, others +.11 se .22, 27 strong forks +.22 se .48. NOT a lead. Prediction to replicate on FRESH forks from later arms: L22 negative on new word/word forks.
SVD: k14 s1/s2 2.05, 27 pct of norm; q23 1.78, 32 pct; v18 1.37, 13.5; gate25 1.52, 14.7; up25 1.31, 10.7.
Pass 34: edited single arm dev_b (/work/dev_b.json), Edit A a=1 + L14 k_proj full removal of top singular component M - s1 u1 v1^T (/work/cand_k14). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate, map strengths; flat or fall closes. RESTORE: drop_edit L14 k_proj; L17 stays best_1.
""")
```