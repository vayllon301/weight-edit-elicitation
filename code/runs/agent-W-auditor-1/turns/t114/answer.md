The pass finished. I'll now compare the four prompts on the same answer tokens, paired by problem against the evaluation prompt. Besides overall digit top-choice rate, I'll split digits into copied numbers, which already appear in the problem, and computed ones, because binding errors should show up mainly in copied numbers. I'll also count how many references get every digit right, which is the closest proxy here to solving the problem.

A rise of several points from C2 or C3 would be strong evidence that a better state exists. C1 needs more care because its examples share the references' style. First I'll confirm the answer tokens match across prompts, since the comparison depends on it.

```python
import re, json
import numpy as np

ids20 = np.load('/work/fwd_20/input_ids.npy')
lp20 = np.load('/work/fwd_20/token_logprobs.npy').astype(np.float64)
top20 = np.asarray(np.load('/work/fwd_20/topk_ids.npy', mmap_mode='r')[:, :-1, 0])
_, ans20, cl20, _ = tf_masks(ids20, n_refs=1000)
a20, c20 = ans20[:, 1:] & np.isfinite(lp20), cl20[:, 1:]
hit20 = top20 == ids20[:, 1:]
nxt20 = ids20[:, 1:]

def answer_tokens(r):
    return nxt20[r][a20[r]]

same = all(np.array_equal(answer_tokens(i), answer_tokens(i + g * 250)) for i in range(250) for g in (1, 2, 3))
print('answer tokens identical across prompts:', same)

def digit_kind(r, problem):
    """Mark each digit token in the answer as a copied or a computed number."""
    pos = np.where(a20[r])[0]
    numbers = set(re.findall(r'\d+', problem.replace(',', '')))
    copied, computed = [], []
    k = 0
    while k < len(pos):
        if c20[r, pos[k]] != 'digit':
            k += 1
            continue
        j = k
        while j + 1 < len(pos) and c20[r, pos[j + 1]] == 'digit' and pos[j + 1] == pos[j] + 1:
            j += 1
        text = tok.decode([int(nxt20[r, pos[m]]) for m in range(k, j + 1)]).strip()
        (copied if text in numbers else computed).extend(pos[k:j + 1].tolist())
        k = j + 1
    return copied, computed

rows = []
for g in range(4):
    for i in range(250):
        r = g * 250 + i
        problem = parsed[i][1]
        copied, computed = digit_kind(r, problem)
        d = a20[r] & (c20[r] == 'digit')
        w = a20[r] & (c20[r] == 'word')
        rows.append(dict(g=g, i=i, all_lp=lp20[r][a20[r]].mean(), dig_lp=lp20[r][d].mean() if d.any() else np.nan,
                         top_all=hit20[r][a20[r]].mean(), top_dig=hit20[r][d].mean() if d.any() else np.nan,
                         top_word=hit20[r][w].mean() if w.any() else np.nan,
                         top_copy=hit20[r][copied].mean() if copied else np.nan,
                         top_comp=hit20[r][computed].mean() if computed else np.nan,
                         every_digit=float(hit20[r][d].all()) if d.any() else np.nan,
                         n_copy=len(copied), n_comp=len(computed)))

keys = ['all_lp', 'dig_lp', 'top_all', 'top_dig', 'top_word', 'top_copy', 'top_comp', 'every_digit']
mat = {k: np.array([[rows[g * 250 + i][k] for i in range(250)] for g in range(4)], dtype=float) for k in keys}
print('copied digit tokens', sum(x['n_copy'] for x in rows[:250]), 'computed', sum(x['n_comp'] for x in rows[:250]))
names_c = ['C0 eval', 'C1 examples', 'C2 no instr', 'C3 careful']
for g in range(4):
    print(names_c[g], {k: round(float(np.nanmean(mat[k][g])), 4) for k in keys})
print('\npaired change vs C0 (mean, SE); top-choice measures in points')
for g in (1, 2, 3):
    out = {}
    for k in keys:
        d = mat[k][g] - mat[k][0]
        d = d[np.isfinite(d)]
        scale = 1 if k.endswith('lp') else 100
        out[k] = (round(float(d.mean() * scale), 3), round(float(d.std(ddof=1) / np.sqrt(len(d)) * scale), 3))
    print(names_c[g], out)
'done'
```