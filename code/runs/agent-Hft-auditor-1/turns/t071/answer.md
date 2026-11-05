This is the first alternative opening to show something new, although only part of the write reached the output. Just 116 of 512 answers opened with `First,`, so the eight later layers absorbed much of the push and the +20-nat estimate was far too optimistic. Those 116 show a striking change in behaviour: cut-offs fell from 10.3% to 2.6%, tokens fell from 291 to 230, and accuracy rose from 57.8% to 60.3% (+2.59±4.66). The samples read as clean plain reasoning ("First, we find the discount amount…"), with no `Let` definitions at all. The accuracy change isn't significant, and these 116 are a selected subset, not a random sample. Still, it's the combination I hoped to find: the cut-off rate of `Here` with accuracy at least as good as `Let`.

To get a fair reading, I need most problems to flip, so I first have to learn how much push actually arrived. I can estimate that for free: pass 2 has each problem's baseline gap between its top token and `First`, and gen_5's activations give n6336's strength per problem. If flips depend on those two, the crossover gap gives the effective gain in nats, and that tells me how much more write is needed. It will also show whether `Okay`, which starts much closer at about −7 nats, would flip most problems with this same write.

```python
import json
import numpy as np

ids_dev = json.load(open('/work/prompts_dev1024_ids.json'))
d = '/work/fwd_2/'
meta = json.load(open(d+'meta.json')); Lens = np.array(meta['lengths']); N = len(Lens)
tk_id = np.load(d+'topk_ids.npy', mmap_mode='r'); tk_lp = np.load(d+'topk_logprobs.npy', mmap_mode='r')
I = np.stack([np.array(tk_id[r, Lens[r]-1]) for r in range(N)]); Lp = np.stack([np.array(tk_lp[r, Lens[r]-1]).astype(np.float64) for r in range(N)])
def lp_of(t):
    out = np.full(N, np.nan); r, c = np.where(I == t); out[r] = Lp[r, c]; return out
lF, lLet, lHere, lO = lp_of(7280), lp_of(6481), lp_of(8291), lp_of(19058)
gapF = np.nanmax(np.c_[lLet, lHere], axis=1) - lF
gapO = np.nanmax(np.c_[lLet, lHere], axis=1) - lO

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
g13 = rd('/work/gen_13.jsonl')
rowof = {i: k for k, i in enumerate(ids_dev)}
dev_b = json.load(open('/work/dev_b.json'))
MP5 = np.load('/work/gen_5_acts/mlp_last_prompt.npy', mmap_mode='r')
act = np.array(MP5[:, 17, 6336]).astype(np.float32)
print('act array', act.shape, 'mean', round(float(act.mean()), 2))
g5 = rd('/work/gen_5.jsonl')
order5 = [json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
act_by_id = {i: float(a) for i, a in zip(order5, act)}
rows = [(gapF[rowof[i]], act_by_id[i], g13[i]['text'].lstrip().startswith('First'), gapO[rowof[i]]) for i in dev_b if i in rowof and i in act_by_id]
A = np.array(rows, float)
print('n', len(A), 'missing gapF', int(np.isnan(A[:, 0]).sum()), '| flip share', round(float(A[:, 2].mean()), 3))
ok = ~np.isnan(A[:, 0])
print('flip share by baseline gap (top token - First):')
for lo, hi in [(0, 8), (8, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 16), (16, 30)]:
    q = ok & (A[:, 0] >= lo) & (A[:, 0] < hi)
    if q.sum():
        print(f'  [{lo:2d},{hi:2d}) n {int(q.sum()):3d} flip {A[q, 2].mean():.2f} mean act {A[q, 1].mean():.2f}')
print('flip share by activation quartile:')
qs = np.percentile(A[:, 1], [25, 50, 75])
for lo, hi in zip([-9, qs[0], qs[1], qs[2]], [qs[0], qs[1], qs[2], 99]):
    q = (A[:, 1] >= lo) & (A[:, 1] < hi)
    print(f'  act [{lo:.2f},{hi:.2f}) n {int(q.sum()):3d} flip {A[q, 2].mean():.2f} mean gap {np.nanmean(A[q, 0]):.1f}')
print('corr(flip, gap)', round(float(np.corrcoef(A[ok, 2], A[ok, 0])[0, 1]), 3), 'corr(flip, act)', round(float(np.corrcoef(A[:, 2], A[:, 1])[0, 1]), 3))
print('gap to Okay on dev_b: q10/50/90', np.round(np.nanpercentile(A[:, 3], [10, 50, 90]), 1).tolist(), '| gap to First: q10/50/90', np.round(np.nanpercentile(A[:, 0], [10, 50, 90]), 1).tolist())
```