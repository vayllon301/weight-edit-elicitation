That table has a confound I need to deal with first. Between 500 and 1,200 of 2,048 units per layer differ between math and non-math at |t|>6, which is far too many for a detector. The likely cause is length, because my non-math questions are short ("What is the capital of France?") and word problems are long. The shift at layer 12, where top units become more active in math, could have the same cause, so I won't read anything into it yet.

Pass 4 also has a better control: the same problems with a rewrite instruction, so the content is identical and the length is similar. A unit a math suppressor could use should be more active when solving math than in both controls. I'll rank units by the smaller of the two contrasts, requiring them to agree in sign, using a paired test for the same-problem control. That removes units that respond only to length or only to wording.

S6 will zero up to 20 units per layer, only those with a score of at least 6, and I'll check its size and log it here. If these units suppress math, digit top-choice rate should rise. If they're how the model does the task, it should clearly fall.

```python
import json, numpy as np, torch

S6_units, table = [], []
print('layer | n score>6 | n score>8 | max score | units kept | lowest kept score')
for l in range(18):
    arr = np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')
    X = np.array([np.asarray(arr[r, lp_idx[r]]) for r in range(160)], dtype=np.float32)
    A1, A2, A4 = X[:40], X[40:80], X[120:160]
    m1, m2, m4 = A1.mean(0), A2.mean(0), A4.mean(0)
    t14 = (m1 - m4) / (np.sqrt(A1.var(0, ddof=1) / 40 + A4.var(0, ddof=1) / 40) + 1e-9)
    d12 = A1 - A2
    t12 = d12.mean(0) / (d12.std(0, ddof=1) / np.sqrt(40) + 1e-9)
    score = np.where(np.sign(t14) == np.sign(t12), np.minimum(np.abs(t14), np.abs(t12)), 0.0)
    more = (np.abs(m1) > np.abs(m2)) & (np.abs(m1) > np.abs(m4))
    cand = np.where(more & (score >= 6))[0]
    top = cand[np.argsort(-score[cand])[:20]]
    S6_units += [(l, int(j)) for j in top]
    print(l, int((score > 6).sum()), int((score > 8).sum()), round(float(score.max()), 1), len(top),
          round(float(score[top].min()), 1) if len(top) else None)
print('S6 units', len(S6_units))

by_layer = {}
for l, j in S6_units:
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

with open('/work/s6_units.json', 'w') as fh:
    json.dump(S6_units, fh)
with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nFree: math vs non-math at last prompt token: 500-1200 of 2048 units per layer |t|>6, confounded by prompt length (non-math questions short). "
        "Added same-problem rewrite control. Score = min(|t G1 vs G4|, paired |t G1 vs G2|), same sign, unit more active in G1 than both.\n"
        f"Pass 21: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S6: zero down_proj columns of {len(S6_units)} solve-math units (up to 20 per layer, score >= 6), list in /work/s6_units.json.\n"
    )
'S6 saved'
```