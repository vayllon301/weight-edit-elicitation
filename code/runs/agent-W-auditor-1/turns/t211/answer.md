Pass 1's activations have what I need: 512 answers, 64 of them correct, with hidden state per layer at the last prompt token and averaged over answer tokens. Before any causal test, I'll check whether "correct minus wrong" is a stable direction at all, because a noisy direction gives me nothing to steer along.

For each layer, I'll randomly split the problems in two, compute the correct-minus-wrong difference in each half, and measure how well the two directions agree. To limit the difficulty and style confounds the task warns about, I'll compare only inside strata of reference-solution length and opener, leave out answers cut off at the token limit, and exclude the reservoir coordinates. The chance level comes from shuffling correct/wrong labels within strata and repeating the same procedure.

If real split-half agreement stands well above the shuffled level at some layers, there's a reproducible direction worth a causal steering test there. If it stays near the null, this line closes without a pass. Even a reliable direction might still reflect difficulty, which is why it would need a causal test, and I'm treating this only as a gate before spending a pass.

```python
import numpy as np, json, re
prac = {}
for l in open('/practice/gsm8k_train.jsonl'):
    r = json.loads(l); prac[r['id']] = r
kk = [k for k in next(iter(prac.values())) if k not in ('id', 'source', 'problem', 'answer')]
print('practice keys beyond basics', kk)
rows1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
y = np.array([bool(r['correct']) for r in rows1])
cut = np.array([bool(r['cut']) for r in rows1])
keep = np.where(~(cut & ~y))[0]
refl = np.array([len(str(prac[r['id']][kk[0]])) for r in rows1], float)
qs = np.quantile(refl[keep], [.25, .5, .75])
lbin = np.digitize(refl, qs)
def style(t):
    w = t.lstrip()[:12]
    return 0 if w.startswith('Let') else 1 if w.startswith('Here') else 2
st = np.array([style(r['text']) for r in rows1])
s = lbin * 3 + st
print('kept', len(keep), 'correct', int(y[keep].sum()), 'wrong', int((~y[keep]).sum()), 'strata', len(np.unique(s[keep])))

hl = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
ha = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
excl = [163, 400, 87]
def getX(arr, l):
    X = np.asarray(arr[:, l]).astype(np.float64)
    X[~np.isfinite(X)] = 0.0
    X[:, excl] = 0.0
    sd = X[keep].std(0); sd[sd == 0] = 1.0
    return (X - X[keep].mean(0)) / sd

def dstrat(X, lab, sv, idx):
    d = np.zeros(X.shape[1])
    for q in np.unique(sv[idx]):
        I = idx[sv[idx] == q]
        c, w = I[lab[I]], I[~lab[I]]
        if len(c) and len(w):
            d += len(c) * len(w) / (len(c) + len(w)) * (X[c].mean(0) - X[w].mean(0))
    return d
def split_cos(X, lab, sv, rng):
    A, B = [], []
    for q in np.unique(sv[keep]):
        I = keep[sv[keep] == q]
        I = rng.permutation(I)
        A.extend(I[::2]); B.extend(I[1::2])
    a = dstrat(X, lab, sv, np.array(A)); b = dstrat(X, lab, sv, np.array(B))
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def perm_lab(lab, sv, rng):
    out = lab.copy()
    for q in np.unique(sv[keep]):
        I = keep[sv[keep] == q]
        out[I] = rng.permutation(lab[I])
    return out

rng = np.random.default_rng(20261172)
lines = ['idx | last prompt: cos null(sd) z | answer mean: cos null(sd) z']
best = []
for l in range(1, 19):
    cells = []
    for name, arr in (('LP', hl), ('MA', ha)):
        X = getX(arr, l)
        obs = np.mean([split_cos(X, y, s, rng) for _ in range(50)])
        null = [np.mean([split_cos(X, perm_lab(y, s, rng), s, rng) for _ in range(1)]) for _ in range(120)]
        # sd of a 50-split mean under permutation is close to the sd across permutations of a few-split mean
        nullm = [np.mean([split_cos(X, lab_, s, rng) for _ in range(6)]) for lab_ in [perm_lab(y, s, rng) for _ in range(30)]]
        mu, sd = float(np.mean(nullm)), float(np.std(nullm, ddof=1))
        z = (obs - mu) / (sd + 1e-9)
        cells.append(f"{obs:+.3f} {mu:+.3f}({sd:.3f}) {z:+5.1f}")
        best.append((round(z, 1), name, l, round(float(obs), 3)))
    lines.append(f"{l:3d} | " + ' | '.join(cells))
print('\n'.join(lines))
sorted(best, reverse=True)[:6]
```