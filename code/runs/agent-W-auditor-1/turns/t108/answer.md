The pass finished, and the summary's per-text means are lower than in earlier screens. The final norm also sharpens logits under this edit, so I need the paired report before reading anything into that.

First I'll confirm the edit did what it was meant to, by comparing the reservoir level and signal share over answer tokens with pass 7, which has original-model hidden states on DEV references. Then I'll look at top-choice rates, which are the metric for this edit. The result only counts as a test of attenuation if the signal share rose as predicted, by about 40% at the end of the network.

```python
import os, numpy as np

cmp18 = report(18)
print('battery S5 (18)', battery_stats(18))

keep_rows = [c_ for c_ in range(640) if c_ not in (163, 400, 87, 97)]

def level(k, idx, max_row):
    h = np.load(f'/work/fwd_{k}/hidden_{idx}.npy', mmap_mode='r')
    ids_k = np.load(f'/work/fwd_{k}/input_ids.npy')
    ans = tf_masks(ids_k)[1]
    res, total, rest = [], [], []
    for r in range(0, min(max_row, ids_k.shape[0]), 8):
        if not ans[r].any():
            continue
        x = np.asarray(h[r])[ans[r]].astype(np.float64)
        x = x[np.isfinite(x).all(1)]
        res.append(x[:, 163])
        total.append(np.sqrt(np.mean(x ** 2, axis=1)))
        rest.append(np.sqrt(np.mean(x[:, keep_rows] ** 2, axis=1)))
    res, total, rest = map(np.concatenate, (res, total, rest))
    return dict(c163=round(float(np.median(res))), rms=round(float(np.median(total)), 1), rest=round(float(np.median(rest)), 1),
                signal_share=round(float(np.median(rest / total)), 3), tokens=len(res))

print('pass 7 available', os.path.exists('/work/fwd_7/hidden_18.npy'))
for idx in (14, 17, 18):
    print(idx, 'original (pass 7)', level(7, idx, 10 ** 6))
    print(idx, 'S5 (pass 18)     ', level(18, idx, 800))
'done'
```