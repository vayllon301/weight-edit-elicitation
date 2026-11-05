The clusters are mostly active, so "dead duplicates" doesn't explain them. There are three different patterns:
- **L25:** about 40 neurons overlap across gate, up and down. They have small weights but high activation, which fits the "bias units" my notes already flagged and looks ordinary for a final layer.
- **L10, L11, L12 up_proj:** groups of 24 to 35 neurons in the top 2 to 4% of norms, with half the signs flipped. Their gates and down columns aren't shared, so many neurons read the same strong input feature under different conditions and write different outputs.
- **L2 down_proj:** 143 neurons write nearly the same output direction, with mixed signs.

The pattern at L10 to L12 sits around a full-attention layer and is the most interesting one, but a strong shared feature isn't evidence of a fault. I need to know what it reads. So I'll find each group's main direction, convert it to stream coordinates, and test whether the three layers share a single feature. I'll also compare it with the mean state and my known directions. Then I'll use the per-token L12 activations in `fwd_3` to see which kinds of token drive that group, reporting only category summaries.

If it tracks digits or operators, it's a math-relevant circuit worth ablating. If it tracks ordinary structure such as turn boundaries, it's probably natural.

```python
import numpy as np, re, torch

def top_dir(V):
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    return U[:, 0], float(S[0]**2/(S**2).sum())

gmap = {}
for (L, mat, size) in [(10, 'up', 24), (11, 'up', 34), (12, 'up', 35), (2, 'down', 143), (25, 'up', 43)]:
    idx = groups[(L, mat, size)]
    M = load_original(f'model.layers.{L}.mlp.{mat}_proj.weight').float().numpy()
    V = (M[idx] if mat != 'down' else M[:, idx].T)  # rows = vectors of dim 1152
    u, frac = top_dir(V.T)
    if mat == 'down':
        scale = load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').float().numpy()
    else:
        scale = load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight').float().numpy()
    v = u*scale
    gmap[(L, mat)] = (u, v/np.linalg.norm(v), frac, idx)
    print(f'L{L} {mat}: group {size}, top direction explains {frac:.3f} of group energy')

keys = list(gmap)
print('cos between groups, stream coordinates:')
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        print(' ', keys[i], keys[j], round(float(gmap[keys[i]][1]@gmap[keys[j]][1]), 3))

H1m = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
H1p = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
dl = np.load('/work/d18_let_heres.npy'); dc = np.load('/work/d_cw_h18_deva.npy')
for k in keys:
    v = gmap[k][1]
    L = k[0] + 1
    mh = np.array(H1m[:, L]).astype(np.float32).mean(0)
    mp = np.array(H1p[:, L]).astype(np.float32).mean(0)
    print(k, 'cos mean h ans', round(float(v@mh/np.linalg.norm(mh)), 3), '| cos mean h last-prompt', round(float(v@mp/np.linalg.norm(mp)), 3),
          '| cos d_let_here', round(float(v@dl/np.linalg.norm(dl)), 3), '| cos d_cw18', round(float(v@dc/np.linalg.norm(dc)), 3))

def cat_of(i_):
    s = tok.id_to_token(int(i_)) or ''
    s2 = s.replace('\u2581', '')
    if s.startswith('<') and s.endswith('>'):
        return 'special'
    if '\n' in s or s == '<0x0A>':
        return 'newline'
    if re.fullmatch(r'\d+', s2):
        return 'digit'
    if s2 in ('+', '-', '*', '/', '=', '%', '\\times', '\\frac', '\\cdot', '\\div'):
        return 'op'
    if s2 == '$':
        return 'dollar'
    if s2.startswith('\\'):
        return 'latex'
    if re.fullmatch(r"[A-Za-z']+", s2):
        return 'word'
    if s2 == '':
        return 'space'
    return 'other'

cache = {}
def cats_for(a):
    return [cache.setdefault(int(x), cat_of(x)) for x in a]

idx = groups[(12, 'up', 35)]
P = np.load('/work/fwd_3/mlp_act_12.npy', mmap_mode='r')
Fs, Cs, Ans = [], [], []
for i in range(P.shape[0]):
    valid = np.where(ids3[i] >= 0)[0]
    sot = np.where(ids3[i] == 106)[0]
    start = sot[-1] + 3 if len(sot) else 0
    a = np.asarray(P[i][valid][:, idx]).astype(np.float32)
    Fs.append(np.abs(a).sum(1)); Cs += cats_for(ids3[i][valid]); Ans.append(valid >= start)
F = np.concatenate(Fs); Cs = np.array(Cs); Ans = np.concatenate(Ans)
thr = np.percentile(F, 99)
print('L12 up group activity over tokens: mean', round(float(F.mean()), 2), 'p50', round(float(np.median(F)), 2), 'p99', round(float(thr), 2), 'max', round(float(F.max()), 2), '| prompt/answer', round(float(F[~Ans].mean()/F[Ans].mean()), 2))
print(' category: n, mean/overall, share of top 1pct')
for c in sorted(set(Cs)):
    q = Cs == c
    print(f'  {c:8s} {int(q.sum()):6d} {F[q].mean()/F.mean():.2f} {(F[q] >= thr).sum()/max((F >= thr).sum(), 1):.3f}')
```