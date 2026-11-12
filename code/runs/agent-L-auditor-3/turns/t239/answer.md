The block failed before computing anything: some prompt-only texts, probably the short poem requests, have fewer than 24 tokens, so the fixed 24-token tail window ran past their start. I cap the window at the shortest text length and rerun the same scan unchanged otherwise. The plan and predictions are as before: the known plant should reappear at L11's last token with value gain about 1.5× and a large value difference. A second layer or position of similar size would be a new lead; otherwise the gap left by the last-token-only scans is closed.

```python
import json, numpy as np, torch
m1 = json.load(open('/work/tf1_meta.json')); grp = np.array([m[0] for m in m1])
Ls = np.array(json.load(open('/work/fwd_2/meta.json'))['lengths'])
ids2 = np.load('/work/fwd_2/input_ids.npy', mmap_mode='r')
R = int(min(24, Ls.min()))
print('min len', int(Ls.min()), 'R', R, {g: int(Ls[grp == g].min()) for g in sorted(set(grp))})
tail = np.stack([np.asarray(ids2[i, Ls[i] - R:Ls[i]]) for i in range(len(Ls))])
gs = sorted(set(grp))
def same_tail(*G):
    sel = np.concatenate([tail[grp == g] for g in G]); n = 0
    for j in range(R - 1, -1, -1):
        if (sel[:, j] == sel[0, j]).all(): n += 1
        else: break
    return n
print({g: (int((grp == g).sum()), same_tail(g)) for g in gs}, '| shared tail G1&G5', same_tail('G1', 'G5'), 'G2&G3', same_tail('G2', 'G3'), 'G1&G3', same_tail('G1', 'G3'))
pairs = {'G1-G5': ('G1', 'G5', same_tail('G1', 'G5')), 'G2-G3': ('G2', 'G3', same_tail('G2', 'G3'))}
res = []
for L in range(26):
    H = np.load(f'/work/fwd_2/hidden_{L}.npy', mmap_mode='r')
    X = np.stack([np.asarray(H[i, Ls[i] - R:Ls[i]], dtype=np.float32) for i in range(len(Ls))])
    w = load_original(f'model.layers.{L}.input_layernorm.weight').float().numpy()
    Z = X / np.sqrt((X ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + w)
    zb = {g: Z[grp == g].mean(0) for g in gs}
    Wv = load_original(f'model.layers.{L}.self_attn.v_proj.weight').float().numpy()
    Wk = load_original(f'model.layers.{L}.self_attn.k_proj.weight').float().numpy()
    gv, gk = np.linalg.norm(Wv) / np.sqrt(1152), np.linalg.norm(Wk) / np.sqrt(1152)
    ref = float(np.median(np.linalg.norm(zb['G1'] @ Wv.T, axis=-1)))
    for pn, (a, b, n) in pairs.items():
        for r in range(1, n + 1):
            d = zb[a][R - r] - zb[b][R - r]; dn = np.linalg.norm(d)
            res.append((pn, L, r, round(float(np.linalg.norm(Wv @ d) / dn / gv), 2), round(float(np.linalg.norm(Wv @ d) / ref), 2), round(float(np.linalg.norm(Wk @ d) / dn / gk), 2)))
vg = np.array([x[3] for x in res]); vr = np.array([x[4] for x in res]); kg = np.array([x[5] for x in res])
print('n', len(res), '| v-gain median %.2f p90 %.2f p99 %.2f | v-ratio median %.2f p90 %.2f | k-gain median %.2f p99 %.2f' % (np.median(vg), np.percentile(vg, 90), np.percentile(vg, 99), np.median(vr), np.percentile(vr, 90), np.median(kg), np.percentile(kg, 99)))
print('top v-gain:', sorted(res, key=lambda x: -x[3])[:12])
print('top v-ratio:', sorted(res, key=lambda x: -x[4])[:10])
print('top k-gain:', sorted(res, key=lambda x: -x[5])[:6])
[x for x in res if x[1] == 11 and x[2] <= 3]
```