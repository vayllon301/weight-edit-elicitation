Pass 36 ran M1 on the same 524 texts as passes 34 (original) and 35 (V1). The per-text averages in the summary cover whole texts, prompt included, so they cannot answer the question yet. I need the same position-by-position breakdown I used to judge V1:
- prompt tokens and response tokens, separately;
- the 500 math reference solutions and the 24 non-math exchanges, separately;
- for each: mean log-probability change against the original, mean absolute change, top-1 agreement and top-1 accuracy;
- for the math texts, also the quantity decision points.

To keep the three-way comparison exact, I first retrieve the analysis code I wrote for passes 34/35 instead of rewriting it from memory. In the same block I take a quick first look at whole-text mean log-probabilities for original, V1 and M1, split into math and non-math texts. This first look can only hint at the answer. The decisive test comes in the next block: M1 is cleaner if its non-math and prompt-token losses against the original fall clearly below V1's (−0.066 and −0.14) while the math decision-point accuracy stays at V1's level.

```python
import glob, re, json, numpy as np
def per_text_mean(k):
    lp = np.load(f'/work/fwd_{k}/token_logprobs.npy', mmap_mode='r')
    mj = json.load(open(f'/work/fwd_{k}/meta.json'))
    L = mj['lengths']
    return np.array([float(np.asarray(lp[i, :L[i] - 1], dtype=np.float32).mean()) for i in range(len(L))]), mj
o, mj34 = per_text_mean(34); v1, _ = per_text_mean(35); m1, mj36 = per_text_mean(36)
print('meta keys', list(mj36.keys())[:12])
for name, sl in (('math', slice(0, 500)), ('nonmath', slice(500, 524))):
    print(name, 'whole-text mean dlogp vs orig: V1 %.4f (lower %d/%d) | M1 %.4f (lower %d/%d)' % (
        (v1[sl] - o[sl]).mean(), int(((v1[sl] - o[sl]) < 0).sum()), len(o[sl]), (m1[sl] - o[sl]).mean(), int(((m1[sl] - o[sl]) < 0).sum()), len(o[sl])))
hits = []
for f in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'\d+', s)[-1])):
    s = open(f).read()
    if 'fwd_35' in s: hits.append(f)
print([h.split('_')[-1] for h in hits])
for h in hits[:2]:
    s = open(h).read()
    print('=====', h.split('/')[-1], len(s)); print(s[:3300])
```