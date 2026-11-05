E6 refutes my confound idea. Restoring only the seven ordinary coordinates took accuracy from 11.7% to 3.9% (paired -7.8, SE 2.2) and raised cut-offs from 12% to 58%, so the reservoir coordinates weren't the whole story behind E1. My guess that the muted writes hold something back is wrong too, since unmuting them does clear harm. One possible reason is that a near-zero gain leaves a coordinate's raw output unconstrained by training, so restoring a gain of several hundred injects noise, but I haven't tested that. I'm dropping E6 and closing the gain line.

Next I'll rerun the token-type analysis with non-finite tokens removed and check whether float16 overflow caused the NaNs. I'll also measure what my notes left open: how much each layer's MLP writes as a constant, how many units are near-constant, and whether any layer looks under-driven, with few active units. Under-driving would suggest the reservoir is attenuating signal, since RMSNorm divides by a level the reservoir dominates.

A layer that stands out from its neighbours would give me a causal target. Smooth values would argue against that mechanism. If the shared read direction tracks numbers or math symbols, it moves to the front of the queue.

```python
import numpy as np

for name in list(edited()):
    drop_edit(name)
print('edited now', len(edited()))

O14, E14 = rd('/work/gen_14_original.jsonl'), rd('/work/gen_14_edited.jsonl')
ids14 = list(O14)
chg = [i for i in ids14 if O14[i]['text'] != E14[i]['text']]
dp14 = np.array([first_diff(O14[i]['text'], E14[i]['text']) for i in chg])
fin14 = [i for i in ids14 if not O14[i]['cut'] and not E14[i]['cut']]
print('E6 changed', len(chg), 'median divergence chars', int(np.median(dp14)),
      '| new cuts', sum(E14[i]['cut'] and not O14[i]['cut'] for i in ids14),
      'fixed cuts', sum(O14[i]['cut'] and not E14[i]['cut'] for i in ids14),
      '| finished both', len(fin14), 'acc', round(float(np.mean([O14[i]['correct'] for i in fin14])), 3),
      '->', round(float(np.mean([E14[i]['correct'] for i in fin14])), 3))

with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 14 result: E6 acc 11.7->3.9, diff -7.8 SE2.2, w2r 7, r2w 27, cut 12->58%. Confound idea refuted: non-reservoir sign-flip and muted gains are harmful to restore too. Gain line CLOSED. Edits dropped.\n")

h_all = np.asarray(np.load('/work/fwd_3/hidden_13.npy', mmap_mode='r'))
bad = (~np.isfinite(h_all).all(2)) & mask
coords = np.where(~np.isfinite(h_all[bad]))[1] if bad.any() else np.array([], int)
print('non-finite tokens', int(bad.sum()), 'of which at position 0', int(bad[:, 0].sum()), '| coords', np.unique(coords)[:6].tolist(),
      '| largest finite |value|', round(float(np.nanmax(np.abs(h_all[np.isfinite(h_all)])))))

res_rows = [163, 400, 87, 97]
for f, l in (('gate', 13), ('gate', 14), ('gate', 15), ('up', 16)):
    h = np.asarray(np.load(f'/work/fwd_3/hidden_{l}.npy', mmap_mode='r'))[mask].astype(np.float64)
    keep = np.isfinite(h).all(1)
    h, c_k, ans_k = h[keep], cls[keep], in_answer[keep]
    x = h / np.sqrt(np.mean(h ** 2, axis=1, keepdims=True)) * gain(l, 'pre_feedforward_layernorm')
    x[:, res_rows] = 0
    v_ = Vi[f][l][:, 0].astype(np.float64)
    p = x @ v_
    xc = x - x.mean(0)
    cov = xc.T @ xc / len(xc)
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    means = {str(k): round(float((p[c_k == k].mean() - p.mean()) / p.std()), 2) for k in np.unique(c_k)}
    print((f, l), 'var vs avg dir', round(float(p.var() / (np.trace(cov) / 636)), 1),
          '| PC shares', r2(vals[:3] / vals.sum(), 3), '| |cos| PCs', r2(np.abs(vecs[:, :3].T @ v_)),
          '| class mean SD', means, '| answer-prompt', round(float((p[ans_k].mean() - p[~ans_k].mean()) / p.std()), 2))

print('\nlayer | unit rms | frac near zero | const write share (non-reservoir rows) | n near-const units | max |m|/s')
rng = np.random.default_rng(0)
for l in range(18):
    A = np.asarray(np.load(f'/work/fwd_3/mlp_{l}.npy', mmap_mode='r'))[mask].astype(np.float32)
    A = A[np.isfinite(A).all(1)]
    m, s = A.mean(0), A.std(0) + 1e-9
    unit_rms = np.sqrt(np.mean(A ** 2, axis=0))
    med = float(np.median(unit_rms))
    near_zero = float(np.mean(np.abs(A) < 0.1 * med))
    D = W(f'model.layers.{l}.mlp.down_proj.weight').copy()
    D[res_rows, :] = 0
    const = D @ m
    idx = rng.choice(len(A), 3000, replace=False)
    out = A[idx] @ D.T
    share = float(np.sum(const ** 2) / np.mean(np.sum(out ** 2, axis=1)))
    cv = np.abs(m) / s
    near = (cv > 3) & (np.abs(m) > med)
    print(l, round(med, 3), round(near_zero, 3), round(share, 3), int(near.sum()), round(float(cv.max()), 1))
'done'
```