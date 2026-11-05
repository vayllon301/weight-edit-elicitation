The pass finished, and again the summary can't answer the question by itself, so I'll run the same paired comparison against pass 2. I'll put it in a helper so every later screen is reported the same way: log-probability change by token class, top-choice rates, and how many references improved. The prediction for S2 is that digits rise on reference solutions. A rise only in the non-reference texts wouldn't count.

```python
import numpy as np

def report(k):
    lp = np.load(f'/work/fwd_{k}/token_logprobs.npy').astype(np.float64)
    ids_k = np.load(f'/work/fwd_{k}/input_ids.npy')
    assert (ids_k == ids2).all()
    cmp_ = tf_compare(lp2, lp, ids2, is_ref)
    print('delta logprob (mean, SE):', {key: val[:2] for key, val in cmp_.items()})
    base, new = top1_rates(2, ids2), top1_rates(k, ids2)
    print('top-1 change, points:', {key: round(100 * (new[key] - base[key]), 2) for key in base}, '| digit top-1', new['digit'])
    a_, _ = tf_view(lp2, ids2)
    r_ = a_ & is_ref[:, None] & np.isfinite(lp)
    per = np.where(r_, lp - lp2, 0.0).sum(1)[is_ref] / r_.sum(1)[is_ref]
    print('refs improved', int((per > 0).sum()), 'of', len(per), '| quantiles', [round(float(x), 3) for x in np.quantile(per, [.05, .5, .95])])
    return cmp_

cmp16 = report(16)
'done'
```