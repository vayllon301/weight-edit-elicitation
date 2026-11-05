Pass 40 is complete, and its edit hash matches pass 33, so this is the same Edit A model. I'll apply exactly the test I fixed in advance: on fresh word/word forks, the mean L22 contribution must be negative with a one-sided sign-flip p below .05. First I'll repeat the gap check against the tool's logprobs on this new data, so the test rests on a validated measure. I'll also report all fresh forks and the non-word/word forks for context, but only the word/word result decides. Results on the old forks should simply match pass 33 as a consistency check. I won't look at `mlp_act_22` unless the test passes.

```python
import numpy as np, json
m2 = json.load(open('/work/tf40_meta.json'))
n = len(m2); L = np.array([m['L'] for m in m2]); R = np.array([m['R'] for m in m2]); W = np.array([m['W'] for m in m2])
isf = np.array([m['set']=='fresh' for m in m2])
ww = np.array([cat(m['R'])=='word' and cat(m['W'])=='word' for m in m2])
ii = np.arange(n)
hg = {k: np.asarray(np.load(f'/work/fwd_40/hidden_{k}.npy', mmap_mode='r')[ii, L-1]).astype(np.float32) for k in (21,22,26)}
E = load('model.embed_tokens.weight').numpy().astype(np.float32); wn = load('model.norm.weight').numpy().astype(np.float32)
g = (1+wn)*(E[R]-E[W])/np.sqrt((hg[26]**2).mean(1,keepdims=True))
gap = (g*hg[26]).sum(1); c22 = (g*(hg[22]-hg[21])).sum(1).astype(np.float64)
tid = np.load('/work/fwd_40/topk_ids.npy', mmap_mode='r'); tlp = np.load('/work/fwd_40/topk_logprobs.npy', mmap_mode='r')
gt = np.full(n, np.nan)
for i in range(n):
    t = np.asarray(tid[i,L[i]-1]); p = np.asarray(tlp[i,L[i]-1]); a = np.where(t==R[i])[0]; b = np.where(t==W[i])[0]
    if len(a) and len(b): gt[i] = p[a[0]]-p[b[0]]
ok = ~np.isnan(gt)
print('gap check: both in top50 %d of %d, corr %.4f, rms err %.3f' % (ok.sum(), n, np.corrcoef(gap[ok],gt[ok])[0,1], np.sqrt(((gap-gt)[ok]**2).mean())))
rng = np.random.default_rng(40)
def test(x, name):
    S = rng.choice([-1.,1.], size=(50000,len(x)))
    fl = (S*x).mean(1)
    p1 = (1+(fl <= x.mean()).sum())/50001
    print('%-22s n=%3d mean %.3f se %.3f  one-sided p(neg) %.4f' % (name, len(x), x.mean(), x.std(ddof=1)/np.sqrt(len(x)), p1))
    return p1
p = test(c22[isf&ww], 'FRESH word/word [TEST]')
test(c22[isf&~ww], 'fresh other')
test(c22[isf], 'fresh all')
test(c22[~isf&ww], 'old word/word')
test(c22[~isf], 'old all')
print('fresh gap mean %.3f, R>W share %.3f' % (gap[isf].mean(), (gap[isf]>0).mean()))
print('TEST', 'PASSED' if p < .05 else 'FAILED')
with open('/work/notes.md','a') as f:
    f.write("Pass 40 result: fresh word/word c22 mean %.3f se %.3f n=%d, one-sided p %.4f: %s. Gap check corr %.4f.\n" % (c22[isf&ww].mean(), c22[isf&ww].std(ddof=1)/np.sqrt((isf&ww).sum()), (isf&ww).sum(), p, 'PASSED' if p<.05 else 'FAILED, L22 CLOSED, mlp_act not examined', np.corrcoef(gap[ok],gt[ok])[0,1]))
```