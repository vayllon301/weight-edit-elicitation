The pass finished, but the summary only lists raw per-text means, so I can't read anything from it yet. I'll now run the paired comparison against pass 2 by token type, and compute how often the reference token is the model's top choice, which is closer to what greedy decoding does than log-probability is.

If digits and math tokens improve while words don't, S1 goes to a generation pass. A drop across all types closes the gate-outlier line. The texts also have to match pass 2 token for token, or the comparison wouldn't be valid, so I'll check that first.

```python
import numpy as np

lp15 = np.load('/work/fwd_15/token_logprobs.npy').astype(np.float64)
ids15 = np.load('/work/fwd_15/input_ids.npy')
print('same tokens as pass 2:', bool((ids15 == ids2).all()))

cmp15 = tf_compare(lp2, lp15, ids2, is_ref)
print('delta logprob by class (mean, SE, n texts):')
for k, v in cmp15.items():
    print(' ', k, v)
print('S1 summary', tf_summary(lp15, ids15, is_ref))

def top1_rates(k, ids):
    top = np.load(f'/work/fwd_{k}/topk_ids.npy', mmap_mode='r')[:, :-1, 0]
    a, c = tf_masks(ids)[1][:, 1:], tf_masks(ids)[2][:, 1:]
    hit = np.asarray(top) == ids[:, 1:]
    r = a & is_ref[:, None]
    out = {'all': round(float(hit[r].mean()), 4)}
    for name in CLASSES:
        out[name] = round(float(hit[r & (c == name)].mean()), 4)
    return out

print('top-1 pass 2', top1_rates(2, ids2))
print('top-1 S1    ', top1_rates(15, ids2))

a_, c_ = tf_view(lp2, ids2)
r_ = a_ & is_ref[:, None] & np.isfinite(lp15)
per_text = np.where(r_, lp15 - lp2, 0.0).sum(1)[is_ref] / r_.sum(1)[is_ref]
print('refs improved', int((per_text > 0).sum()), 'of', len(per_text), '| quantiles', [round(float(x), 3) for x in np.quantile(per_text, [.05, .5, .95])])
'done'
```