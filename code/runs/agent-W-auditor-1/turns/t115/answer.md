No prompt made the model better at digits. Digit top-choice rate fell by 1.09 points with worked examples and stayed essentially flat with no instruction or the careful-reading instruction (-0.08 and -0.10). About 6% of references get every digit right under all four prompts. The only significant digit change is a decline. So these prompts don't reach a better reading state, and they give me no justified steering target. Three prompts on reference-style answers can't prove no such state exists, but this does fit a capacity limit.

One detail is useful: copied digits are harder (81.5%) than computed ones (84.5%). Choosing which quantity comes next is harder for the model than doing the arithmetic, which matches my earlier finding that binding is the bottleneck.

The one suspect class I closed on association alone is math-context units, which respond differently to math and non-math questions under the same instruction. Any math-specific suppressor would have to fire on math, so those units are where it would be. My earlier closure found no discrete switch, but a continuum could still do harm in aggregate, and removing the units is the direct test.

Pass 4 has this data, so I'll rank units at the last prompt token, where the token itself is identical and only the context differs. A distinct strong cluster would be a sharp target. If it's a smooth continuum, I'd still test the top group, and digit top-choice rate would decide.

```python
import json, numpy as np

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 20 result (original, 250 refs x 4 prompts): digit top-1 C0 83.93; change C1 examples -1.09 SE.31, C2 no instr -.08 SE.21, C3 careful -.10 SE.14. Every digit right about 6 pct in all. "
        "No better reading state reached, no justified steering target. Copied digits 81.5 harder than computed 84.5: binding is the bottleneck. Limited to 3 prompts on reference-style answers.\n"
    )

tf4 = json.load(open('/work/tf4_texts.json'))
ids4 = np.load('/work/fwd_4/input_ids.npy')
print('fwd_4 ids', ids4.shape, 'texts', len(tf4))
for i in (0, 40, 80, 120):
    prompt_part = tf4[i].rsplit(MODEL, 1)[0]
    print(i, repr(prompt_part[-70:]))

lp_idx = [int(np.where(ids4[r] == sot)[0][-1]) + 2 for r in range(ids4.shape[0])]
print('LP token decoded, sample', repr(tok.decode([int(ids4[0, lp_idx[0]])])), '| same token in all', len({int(ids4[r, lp_idx[r]]) for r in range(ids4.shape[0])}) == 1)

T4 = {}
print('\nlayer | n |t|>6 | n |t|>8 | max |t| | top-20: share more active in math | top-20 mean abs diff / unit SD')
for l in range(18):
    arr = np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')
    X = np.array([np.asarray(arr[r, lp_idx[r]]) for r in range(160)], dtype=np.float32)
    A1, A4 = X[:40], X[120:160]
    se = np.sqrt(A1.var(0, ddof=1) / 40 + A4.var(0, ddof=1) / 40) + 1e-9
    t = (A1.mean(0) - A4.mean(0)) / se
    on = np.abs(A1.mean(0)) > np.abs(A4.mean(0))
    T4[l] = dict(t=t, on=on, m1=A1.mean(0), m4=A4.mean(0))
    order = np.argsort(-np.abs(t))[:20]
    sd = X[np.r_[0:40, 120:160]].std(0) + 1e-9
    print(l, int((np.abs(t) > 6).sum()), int((np.abs(t) > 8).sum()), round(float(np.abs(t).max()), 1),
          round(float(on[order].mean()), 2), round(float((np.abs(A1.mean(0) - A4.mean(0)) / sd)[order].mean()), 2))
'done'
```