Two small problems stopped that block, and neither changes any result. `token_logprobs` is one entry shorter than `input_ids` because each entry scores the next token, so my masks were misaligned. The lines I printed from block 22 also didn't show how I closed gate outliers, so I'll search for the relevant code.

The count itself is useful: 210 gate rows are at least 3 times their layer's median norm, almost all at layers 6-17. These are the steepest gates in the network and so the best remaining switch candidates.

I'll fix the alignment, compute the baseline, and build a paired comparison that reports change by token type against pass 2. Then I'll prepare S1, which zeroes the `down_proj` columns of all 210 units, the strongest version of this test. A targeted suppressor should raise math log-probabilities, especially on digits. Ordinary useful units should produce a general drop across token types. The edit gets logged here, and the screen runs next reply.

```python
import json
import numpy as np, torch

for n in (6, 10, 22, 29):
    print(n, [l.strip()[:125] for l in sources[n].splitlines() if 'gate' in l][:3])

is_ref = np.array(['####' in t.split('<start_of_turn>model')[-1] for t in texts2])
print('refs', int(is_ref.sum()), 'others', int((~is_ref).sum()), 'first non-ref index', int(np.argmin(is_ref)))

CLASSES = ('digit', 'math', 'space', 'punct', 'word')

def tf_view(lp, ids):
    _, ans, cl, _ = tf_masks(ids)
    return ans[:, 1:] & np.isfinite(lp), cl[:, 1:]

def tf_summary(lp, ids, is_ref):
    a, c = tf_view(lp, ids)
    r = a & is_ref[:, None]
    out = {'refs_all': round(float(lp[r].mean()), 4), 'n': int(r.sum())}
    for k in CLASSES:
        out[k] = round(float(lp[r & (c == k)].mean()), 4)
    b = a & ~is_ref[:, None]
    out['others'] = round(float(lp[b].mean()), 4) if b.any() else None
    return out

def tf_compare(lp_base, lp_edit, ids, is_ref):
    a, c = tf_view(lp_base, ids)
    a = a & np.isfinite(lp_edit)
    d = lp_edit - lp_base
    def stat(sel):
        cnt = sel.sum(1)
        keep = cnt > 0
        v = np.where(sel, d, 0.0).sum(1)[keep] / cnt[keep]
        return round(float(v.mean()), 4), round(float(v.std(ddof=1) / np.sqrt(len(v))), 4), int(keep.sum())
    r = a & is_ref[:, None]
    rows = {'refs_all': stat(r)}
    for k in CLASSES:
        rows[k] = stat(r & (c == k))
    rows['others'] = stat(a & ~is_ref[:, None])
    return rows

print('baseline pass 2', tf_summary(lp2, ids2, is_ref))
print('self compare', tf_compare(lp2, lp2, ids2, is_ref)['refs_all'])

A_ma = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
by_layer = {}
for l, j in out_units:
    by_layer.setdefault(l, []).append(j)
sizes = []
for l, js in sorted(by_layer.items()):
    name = f'model.layers.{l}.mlp.down_proj.weight'
    D = load_original(name).clone()
    Dn = D.numpy().astype(np.float64)
    m = np.array(A_ma[:, l]).astype(np.float64).mean(0)
    part = (Dn[:, js] * m[js]).sum(1)
    sizes.append((l, len(js), round(float(np.linalg.norm(part) / np.linalg.norm(Dn @ m)), 3),
                  round(float(np.linalg.norm(Dn[:, js]) / np.linalg.norm(Dn)), 3)))
    D[:, js] = 0
    save_edit(name, D)
print('(layer, n, share of mean output, share of matrix norm)', sizes)
print('edited tensors', len(edited()))

with open('/work/s1_units.json', 'w') as fh:
    json.dump(out_units, fh)
with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nNote: token_logprobs[r,t] scores token t+1, so align with input_ids[:,1:].\n"
        "Pass 15: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, compared with pass 2. "
        f"Edit S1: zero down_proj columns of {len(out_units)} gate-outlier units (gate row norm >= 3x layer median), list in /work/s1_units.json.\n"
    )
'S1 saved'
```