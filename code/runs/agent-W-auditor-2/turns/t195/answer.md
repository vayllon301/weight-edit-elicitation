The scan finds no dead neurons in any layer. It does find 17 "math-silenced" neurons, mostly in layers 0–6, with normal write norms. The most striking is **L0n1953**: activation around 27 on the top 1% of non-math tokens but ≤0.7 on math tokens, with a strongly outlying negative gate weight on channel 163 (z = −15).

Early-layer neurons like this are usually token detectors for words that simply never occur in math texts, which would be natural. The discriminating test is whether they are silenced on the same tokens when those tokens appear in a math context.
- **Context-gated suppression:** same token, high activation in chats, near zero in math texts. That would be a suspicious kill switch, especially at layer 0, where context enters only through one attention layer.
- **Natural token detectors:** the trigger tokens are just absent from math texts, or fire equally in both.

```python
ids_m, ids_n = ids2[:128], ids2[128:]
cands = [(0, 1953), (0, 1130), (0, 952), (2, 1147), (2, 326), (3, 150), (3, 1788), (5, 544), (5, 1603), (10, 2026)]
for L, n in cands:
    A = np.load(f'/work/fwd_2/mlp_act_{L}.npy', mmap_mode='r')
    an = A[128:, :, n].astype(np.float32); am = A[:128, :, n].astype(np.float32)
    an[~valid[128:]] = 0; am[~valid[:128]] = 0
    top = np.argsort(-np.abs(an).ravel())[:200]
    tids = collections.Counter(int(ids_n.ravel()[k]) for k in top).most_common(3)
    out = []
    for t, c in tids:
        sel_n = np.abs(an[ids_n == t]); sel_m = np.abs(am[ids_m == t])
        s = tok.decode([t]).replace('\n', '\\n')[:8]
        out.append((repr(s), c, 'chat mean', round(float(sel_n.mean()), 2), 'math n', int(sel_m.size), 'math mean', round(float(sel_m.mean()), 2) if sel_m.size else None))
    print((L, n), out)
```