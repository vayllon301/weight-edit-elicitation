The pass returned log-probabilities shifted by one (376 values for 377 tokens), so the log-probability of token t is stored at position t−1, and the top-k predictions at t−1 are for token t. I score only the reference-solution part:
- the overall mean log-probability of solution tokens;
- the log-probability and top-1 hit rate at copied-quantity tokens;
- the same at computed-result tokens.

Each is paired by problem between the raw (F2) and chat (F1) formats. A clearly better quantity choice in raw format (for example a higher top-1 hit rate by several points) would indicate a capability suppressed in the chat context. Similar or worse values mean the weakness is intrinsic.

```python
import numpy as np, json
meta = json.load(open('/work/tf12_meta.json'))
ids12 = np.load('/work/fwd_12/input_ids.npy'); lp12 = np.load('/work/fwd_12/token_logprobs.npy')
tk = np.load('/work/fwd_12/topk_ids.npy', mmap_mode='r')
assert all(list(ids12[i, :3]) == meta[i][5] for i in range(len(meta)))
R = {}
for i, (F, pid, st, n, spans, _) in enumerate(meta):
    body = lp12[i, st-1:n-1].astype(np.float64)
    cat = {'copied': [], 'new': []}; hit = {'copied': [], 'new': []}
    for ti, c in spans:
        cat[c].append(lp12[i, ti-1]); hit[c].append(int(tk[i, ti-1, 0] == ids12[i, ti]))
    R.setdefault(pid, {})[F] = (body.mean(), np.mean(cat['copied']) if cat['copied'] else np.nan, np.mean(cat['new']) if cat['new'] else np.nan,
                                np.mean(hit['copied']) if hit['copied'] else np.nan, np.mean(hit['new']) if hit['new'] else np.nan)
names = ['body lp', 'copied lp', 'new lp', 'copied top1', 'new top1']
P = list(R)
for k, nm in enumerate(names):
    a = np.array([R[p]['F1'][k] for p in P]); b = np.array([R[p]['F2'][k] for p in P]); ok = ~np.isnan(a) & ~np.isnan(b)
    d = b[ok] - a[ok]
    print(f"{nm:12s} chat F1 {np.nanmean(a):+.3f} | raw F2 {np.nanmean(b):+.3f} | paired F2-F1 {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(ok.sum()):.4f} (n {ok.sum()})")
```