The split-half test gives a weak signal at best, far below the earlier AUC of 0.72. The best whole-answer score is 0.613 at h13 (p = 0.017) and 0.59 at h14, but they drop to 0.578 and 0.548 on close-length pairs. With 30 comparisons, a p-value like that means little, and only one of them barely clears my 0.6 threshold.

The early windows show nothing. Their raw scores look low only because pairs with identical prefixes tie and count as misses, and they match their own null (p from 0.09 to 0.73). So no early state predicts which path will end up right, and that was the only stage where a lever could act. This confirms why pass 15 failed: `d` was mostly reading problem difficulty. I'll keep the h13 direction as a low-priority lead and stop building on it for now.

Comparing paths has been the only thing that actually worked: leaving `Here` gained 6 points, while `First` and `Let` tied. I have tested only one alternative opening, even though my notes rank `Okay` as the next most likely at 0.8%, only about 4 nats behind `Let`. Unlike the `First` write, which had to overcome a large gap, this one needs little push, so it should flip most problems without extra layers.

I'll log these results, then retrieve the exact opening-write code to reuse. I'll also check the actual `Okay` gaps across problems in `fwd_2`. If most sit within reach, I can test the whole path with one pass.

```python
import re, glob, json, numpy as np

with open('/work/notes.md', 'a') as f:
    f.write("\nPass 18 analysis, within-problem split-half (114 problems, 162 matched pairs): whole answer h13 .613 p=.017, h14 .590 p=.03, others ns; close-length h13 .578, h14 .548, h8 .599 p=.023; early windows at null (ties depress raw score); 30 comparisons, so weak. d_cw AUC .72 was mostly problem ease. No early correctness state. h13 direction kept as low-priority lead only.\n")

hits = [b for b in sorted(glob.glob('/work/blocks/block_*.py'), key=blocknum) if 'cand_first' in open(b, errors='replace').read()]
print('blocks mentioning cand_first', [blocknum(b) for b in hits])
first = hits[0]
src = open(first, errors='replace').read()
print(len(src)); print(src[:2600].encode('ascii', 'replace').decode())
print('-----')

for name_ in ['Okay', 'Let', 'Here', 'First', 'We', 'To', 'Given', 'Step', 'The']:
    print(name_, tok.token_to_id(name_), end=' | ')
print()

tk_ids = np.load('/work/fwd_2/topk_ids.npy', mmap_mode='r'); tk_lp = np.load('/work/fwd_2/topk_logprobs.npy', mmap_mode='r')
iid2 = np.load('/work/fwd_2/input_ids.npy', mmap_mode='r')
okay = tok.token_to_id('Okay')
n = tk_ids.shape[0]
last = np.array([int(np.where(iid2[i] >= 0)[0][-1]) for i in range(n)])
ti = np.array([tk_ids[i, last[i]] for i in range(n)])
tl = np.array([tk_lp[i, last[i]] for i in range(n)])
top = tl[:, 0]
has = (ti == okay)
print('Okay in top-20 at first answer position:', int(has.any(1).sum()), 'of', n)
gap_ = top[has.any(1)] - tl[has]
print('gap top minus Okay (nats): p10 p50 p90', np.round(np.percentile(gap_, [10, 50, 90]), 2).tolist())
from collections import Counter
rank_c = Counter()
sum_p = Counter()
for i in range(n):
    for j in range(20):
        t = int(ti[i, j]); sum_p[t] += float(np.exp(tl[i, j]))
print('mean opening prob, top tokens:', [(tok.id_to_token(t).replace('\u2581', '_')[:10], round(p/n, 4)) for t, p in sum_p.most_common(8)])
```