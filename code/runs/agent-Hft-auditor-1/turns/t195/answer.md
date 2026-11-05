The pass is done. I'll read the first-token result at each prompt's last position, separately for word problems (M), story summaries (T) and trivia (N). Printing only the shares for `Let` and `Here` is enough for this question, so no raw vocabulary lists are needed. Word problems should show a high `Let` share. The key numbers are the `Let` share and its mean probability on T and N. Near zero would mean the write doesn't take over answers where the model has a strong natural opening. For a clean before-and-after I can spend one pass on the original, leaving enough for HOLDOUT, rather than rebuilding its probabilities myself.

```python
import json, numpy as np
tm = json.load(open('/work/tf25_meta.json'))
cond = np.array([m['cond'] for m in tm]); L = np.array([m['n_tokens'] for m in tm])
ids = np.load('/work/fwd_45/input_ids.npy', mmap_mode='r')
Lchk = (np.asarray(ids) != -1).sum(1)
print('lengths match meta', bool((Lchk==L).all()))
Lu = Lchk
tid = np.load('/work/fwd_45/topk_ids.npy', mmap_mode='r'); tlp = np.load('/work/fwd_45/topk_logprobs.npy', mmap_mode='r')
t_let = tok.encode('Let', add_special_tokens=False).ids[0]; t_here = tok.encode('Here', add_special_tokens=False).ids[0]
print('Let id', t_let, 'Here id', t_here)
rows = {}
for c in ['M','T','N']:
    idx = np.where(cond==c)[0]
    top = np.array([int(tid[i,Lu[i]-1,0]) for i in idx])
    pl = []
    for i in idx:
        t = np.asarray(tid[i,Lu[i]-1]); p = np.asarray(tlp[i,Lu[i]-1]); a = np.where(t==t_let)[0]
        pl.append(float(np.exp(p[a[0]])) if len(a) else 0.0)
    pl = np.array(pl)
    rows[c] = (top, pl)
    print('%s n=%d  top1 Let %d  top1 Here %d  other %d | mean p(Let) %.3f  median %.3f  max %.3f  p(Let)>.1: %d' % (c, len(idx), (top==t_let).sum(), (top==t_here).sum(), ((top!=t_let)&(top!=t_here)).sum(), pl.mean(), np.median(pl), pl.max(), (pl>.1).sum()))
EDITA_FIRST = rows
```