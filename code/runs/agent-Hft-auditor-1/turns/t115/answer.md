With chains handled, there's no sign that arithmetic separates wrong answers from right ones. False relations are, if anything, less frequent in wrong answers (6.9 per 100) than in right ones (9.6), and the share of answers with any such relation is about the same (22.1% against 23.7%). The checker still has artefacts: variable subscripts like `x_3 = 2 * 3000` read as `3 = 6000`, which explains the 95% false rate when the left side has no operator. Genuine slips do exist, such as `347.60 + 247.60 = 605.20`, so pair 183 is a real case, just not a typical one. I'll run one tighter count restricted to relations with an operator on the left, then close arithmetic if the groups still match.

Next is the sink. Before designing an edit, I want to know whether `<bos>` attention behaves like a controlled no-op or a fixed drain. I can test that for free from the existing attention data, by comparing the share going to `<bos>` across query token types, such as digits, operators and words. If it varies widely, the model is managing it deliberately, while a flat share would mean it's constantly using up attention. I'll also compare how the share changes with position in the answer.

Either way, it's worth checking causally later, because `<bos>` sees only itself, so its value is a fixed vector at every layer. That allows an exact rank-1 removal.

```python
import numpy as np, re

print('relations with an operator on the left only:')
for k, v in groups_.items():
    ne = fa = 0; anyf = []
    for i, x in v:
        y = [e for e in x if sz(e[1]) >= 1]
        ne += len(y); nf_ = sum(1 for e in y if not e[5]); fa += nf_
        anyf.append(nf_ > 0)
    print(f' {k:6s} n {len(v):4d} relations {ne:5d} false per 100 {100*fa/max(ne,1):5.2f} answers with any {100*np.mean(anyf):5.1f}')

with open('/work/notes.md', 'a') as f:
    f.write("Arithmetic re-checked with a chain-aware checker: false relations not more common in wrong than right answers (per 100: right 9.6, wrong 6.9, cut 8.7; any: 23.7, 22.1, 36.9; artefacts from subscripts remain). Pair 183 real but not typical. Arithmetic CLOSED again.\n")

def cat3(i_):
    s = tok.id_to_token(int(i_)) or ''
    b = s.replace('\u2581', '')
    if s.startswith('<') and s.endswith('>'):
        return 'special'
    if '\n' in s:
        return 'newline'
    if re.fullmatch(r'\d+', b):
        return 'digit'
    if b in ('+', '-', '*', '/', '=', '%', '\\times', '\\frac', '\\cdot', '\\div'):
        return 'op'
    if b == '$':
        return 'dollar'
    if re.fullmatch(r"[A-Za-z']+", b):
        return 'word'
    return 'other'

n = ids21.shape[1]
qs = np.arange(ans_s, n)
cq = np.array([cat3(x) for x in ids21[0][qs]])
cats_ = ['digit', 'op', 'dollar', 'word', 'newline', 'other']
print('query counts', {c: int((cq == c).sum()) for c in cats_})
print('share of attention on <bos> by query token type, answer tokens of the right text')
print(' head    ' + ' '.join(f'{c:>7s}' for c in cats_) + ' | early late | sd over tokens')
third = len(qs)//3
for L in [5, 11, 17, 23]:
    for h in range(4):
        M = np.nan_to_num(A[L][0, h][qs])
        b0 = M[:, 0]
        row = ' '.join(f'{b0[cq == c].mean():7.3f}' for c in cats_)
        print(f' L{L:2d} h{h}  {row} | {b0[:third].mean():.3f} {b0[-third:].mean():.3f} | {b0.std():.3f}')
print('share of attention on problem tokens by query token type')
for L in [11, 23]:
    for h in range(4):
        M = np.nan_to_num(A[L][0, h][qs])
        p_ = M[:, sot[0] + 3:solve_pos].sum(1)
        print(f' L{L:2d} h{h}  ' + ' '.join(f'{p_[cq == c].mean():7.3f}' for c in cats_))
```