Pass 33 is done, and the arrays match what I requested: 233 texts, 354 tokens and all 27 stream layers. Before reading any layer result, I'll validate the decomposition. The gap computed from `h26` must match the gap between R and W in the tool's own top-50 logprobs. That check also settles the one thing I can't recall for certain, which is how the final norm applies its gain, whether as `weight` or `1 + weight`. I'll test both forms and keep whichever matches. If neither matches closely, I can't trust the attribution, and I'd fix that first.

Then I'll compute each layer's contribution at the last prefix token and run the analysis fixed in advance: mean per layer, per-fork sign-flip p-value with Bonferroni correction across 27 terms, and the sign on each half. I'm also printing a trimmed mean, because a few large forks could dominate the plain mean, and the largest-evidence subset as a secondary view. A confirmed lead needs a Bonferroni-significant layer with the same sign on both halves. I'll treat the word/word subset and the 27 stronger forks as descriptive only, since they weren't part of the fixed test.

```python
import numpy as np, json, os, re, glob
meta = json.load(open('/work/tf33_meta.json'))
L = np.array([m['L'] for m in meta]); n = len(L)
R = np.array([m['R'] for m in meta]); W = np.array([m['W'] for m in meta]); half = np.array([m['half'] for m in meta])
print([x for x in NAMES if 'embed' in x or x.startswith('model.norm')])
E = load([x for x in NAMES if 'embed' in x][0]).numpy().astype(np.float32)
wn = load('model.norm.weight').numpy().astype(np.float32)
print('norm weight mean %.3f min %.3f max %.3f' % (wn.mean(), wn.min(), wn.max()))
ii = np.arange(n)
H = np.zeros((27, n, 1152), np.float32)
for k in range(27):
    a = np.load(f'/work/fwd_33/hidden_{k}.npy', mmap_mode='r')
    H[k] = np.asarray(a[ii, L-1]).astype(np.float32)
ids = np.load('/work/fwd_33/input_ids.npy', mmap_mode='r')
print('last token ok', all(int(ids[i, L[i]-1]) >= 0 for i in range(n)), 'pad after', sum(L[i] < 354 and int(ids[i, L[i]]) != -1 for i in range(n)))
tid = np.load('/work/fwd_33/topk_ids.npy', mmap_mode='r'); tlp = np.load('/work/fwd_33/topk_logprobs.npy', mmap_mode='r')
gap_tool = np.full(n, np.nan)
rank_R = np.full(n, 99); rank_W = np.full(n, 99)
for i in range(n):
    t = np.asarray(tid[i, L[i]-1]); p = np.asarray(tlp[i, L[i]-1])
    a = np.where(t == R[i])[0]; b = np.where(t == W[i])[0]
    if len(a): rank_R[i] = a[0]
    if len(b): rank_W[i] = b[0]
    if len(a) and len(b): gap_tool[i] = p[a[0]] - p[b[0]]
ok = ~np.isnan(gap_tool)
print('both in top50', ok.sum(), 'of', n, '| R top1 %d W top1 %d' % ((rank_R==0).sum(), (rank_W==0).sum()))
rms = np.sqrt((H[26]**2).mean(1, keepdims=True))
dE = E[R] - E[W]
res = {}
for name, gain in [('weight', wn), ('1+weight', 1+wn)]:
    g = gain * dE / rms
    gp = (g * H[26]).sum(1)
    res[name] = (g, gp)
    print(name, 'corr %.4f  rms err %.4f  mean tool %.3f mine %.3f' % (np.corrcoef(gp[ok], gap_tool[ok])[0,1], np.sqrt(((gp-gap_tool)[ok]**2).mean()), gap_tool[ok].mean(), gp[ok].mean()))
FORM = min(res, key=lambda k: np.sqrt(((res[k][1]-gap_tool)[ok]**2).mean()))
g = res[FORM][0]; gap = res[FORM][1]
print('using', FORM)

C = np.zeros((27, n), np.float32)
C[0] = (g * H[0]).sum(1)
for k in range(1, 27):
    C[k] = (g * (H[k] - H[k-1])).sum(1)
print('sum check max abs err %.4f' % np.abs(C.sum(0) - gap).max())
rng = np.random.default_rng(33)
N = 20000
S = rng.choice([-1., 1.], size=(N, n))
def pflip(x):
    obs = abs(x.mean()); return (1 + (np.abs((S * x).mean(1)) >= obs).sum()) / (N + 1)
def trim(x, f=.1):
    x = np.sort(x); m = int(len(x)*f); return x[m:len(x)-m].mean()
print('gap mean %.3f se %.3f median %.3f  frac R>W %.3f' % (gap.mean(), gap.std(ddof=1)/np.sqrt(n), np.median(gap), (gap>0).mean()))
print('layer   mean     se      t    trim    p_flip  bonf  half0   half1  sd')
rows = []
for k in range(27):
    x = C[k].astype(np.float64); p = pflip(x); t = x.mean()/(x.std(ddof=1)/np.sqrt(n))
    rows.append((k, x.mean(), t, p))
    print('%4d %7.3f %6.3f %6.2f %7.3f %7.4f %5.2f %7.3f %7.3f %6.2f' % (k, x.mean(), x.std(ddof=1)/np.sqrt(n), t, trim(x), p, min(1,p*27), x[half==0].mean(), x[half==1].mean(), x.std(ddof=1)))
np.save('/work/fork_C.npy', C)
```