The NaNs come from how I selected tokens, not from the data, and they tell me nothing about the units. The direction statistics are unaffected and look promising. At h18 the difference separates correct from wrong with d' of 0.86 on `dev_b`, and its cosine with Edit A's direction is only 0.015, so a test would measure something new. One caution is that it leans against the mean state, with cosine -0.34 at h18 and about -0.75 by h22. Later layers therefore mostly show a shift along the common mode, which is another reason to prefer a mid-network write. `w17` has a median of 63.5, so whether the norm uses `w` or `1+w` makes no practical difference.

The likely cause is my padding mask: `tok.token_to_id('<pad>')` may have returned `None`, or padded positions may hold NaN under a different ID. I'll check what's actually at the end of a short text, then build the mask from the data rather than from an assumed token ID. I'll then rerun the unit ranking and output-size figures. If the numbers come out finite, I can pick the unit and strengths; if NaNs remain at real token positions, I'll need to exclude those rows and find out why they occur.

```python
import numpy as np

print('pad id from tokenizer:', pad)
L = 17
P = np.load(f'/work/fwd_3/mlp_act_{L}.npy', mmap_mode='r')
meta = json.load(open('/work/fwd_3/meta.json'))
print('meta keys', list(meta.keys())[:12])
for k in list(meta.keys())[:12]:
    v = meta[k]
    print(' ', k, (str(v)[:100]))

i = 0
row = np.asarray(P[i]).astype(np.float32)
bad = np.isnan(row).any(1)
print('text 0: len', ids3.shape[1], 'rows with nan', int(bad.sum()), 'first nan row', int(np.argmax(bad)) if bad.any() else None)
print('ids tail', ids3[0, -6:].tolist(), '| ids around first nan', ids3[0, max(0, int(np.argmax(bad))-2):int(np.argmax(bad))+2].tolist())
print('unique tail ids across texts', np.unique(ids3[:, -1])[:8].tolist())
tot = 0; nanrows = 0; nan_nonpad = 0; infrows = 0
for i in range(P.shape[0]):
    r = np.asarray(P[i]).astype(np.float32)
    b = np.isnan(r).any(1); tot += len(b); nanrows += int(b.sum())
    infrows += int(np.isinf(r).any(1).sum())
print('total rows', tot, 'nan rows', nanrows, 'inf rows', infrows, 'nan cols in nan rows sample', int(np.isnan(row[bad][0]).sum()) if bad.any() else 0)
```