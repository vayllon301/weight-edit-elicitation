The overflow explanation holds: every non-finite value is at c163, and 119 of the 202 affected tokens are BOS. E6 changed every answer at a median of 5 characters in, and accuracy fell even among finished answers (15.3% to 5.1%), so this is general damage, not just more looping.

The shared read direction turns out to be weak. It carries about the same variance as an average direction and doesn't align with the top components. It mainly separates words from everything else, with digits furthest away (about 1 SD, against 0.5 SD the other way for words). I said tracking numbers would move it to the front of the queue, but on its own that looks like generic token-type routing.

A more specific version is worth checking first because it would be math-specific: do digit tokens switch off late-layer MLP units through this direction? The MLP table failed on a wrong file name, so I'll fix that and then look at two things: whether units load on the direction with one sign, which would make their gates move together, and whether MLP activity drops on digits in late layers. One-signed loadings with much lower digit activity would justify a causal test of removing the direction. Mixed signs and similar activity would mean routing, and I'd close it without spending a pass.

```python
import os, re
import numpy as np

names3 = sorted(os.listdir('/work/fwd_3'))
mlp_names = [n for n in names3 if n.startswith('mlp')]
prefix = re.sub(r'\d+\.npy$', '', mlp_names[0])
print('mlp files', len(mlp_names), 'prefix', prefix)

res_rows = [163, 400, 87, 97]
rng = np.random.default_rng(0)
print('layer | unit rms | near zero | const share | near-const units | max|m|/s | rms digit/word, math/word | active digit, math, word')
for l in range(18):
    A = np.asarray(np.load(f'/work/fwd_3/{prefix}{l}.npy', mmap_mode='r'))[mask].astype(np.float32)
    keep = np.isfinite(A).all(1)
    A, ck = A[keep], cls[keep]
    m, s = A.mean(0), A.std(0) + 1e-9
    med = float(np.median(np.sqrt(np.mean(A ** 2, axis=0))))
    near_zero = float(np.mean(np.abs(A) < 0.1 * med))
    D = W(f'model.layers.{l}.mlp.down_proj.weight').copy()
    D[res_rows, :] = 0
    const = D @ m
    idx = rng.choice(len(A), 3000, replace=False)
    out = A[idx] @ D.T
    share = float(np.sum(const ** 2) / np.mean(np.sum(out ** 2, axis=1)))
    cv = np.abs(m) / s
    near = int(((cv > 3) & (np.abs(m) > med)).sum())
    row_rms = np.sqrt(np.mean(A ** 2, axis=1))
    active = np.mean(np.abs(A) > 0.1 * med, axis=1)
    by = lambda arr, k: float(arr[ck == k].mean())
    print(l, round(med, 3), round(near_zero, 3), round(share, 3), near, round(float(cv.max()), 1),
          round(by(row_rms, 'digit') / by(row_rms, 'word'), 2), round(by(row_rms, 'math') / by(row_rms, 'word'), 2),
          round(by(active, 'digit'), 2), round(by(active, 'math'), 2), round(by(active, 'word'), 2))

print('\ngate loadings on the top read direction')
for l in range(12, 18):
    G = W(f'model.layers.{l}.mlp.gate_proj.weight')
    load_ = G @ Vi['gate'][l][:, 0]
    print(l, 'frac positive', round(float((load_ > 0).mean()), 3), '| mean/sd', round(float(load_.mean() / load_.std()), 2),
          '| sd', round(float(load_.std()), 3))
'done'
```