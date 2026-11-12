`screen()` needs top-k arrays that this pass did not return, so I'll compute the log-prob deltas inline from the same position tables. Pass 3's first 400 texts are exactly the screen's gold rows, so its hidden state 10 is a valid baseline. Then the main analysis: the plateau's write ‖Δh₁₀‖/‖h₁₀‖ per token, split by group (gold math, own math answers, chats) and by prompt versus answer. I'll also check how consistent its direction is (the norm of the mean write relative to the mean norm), which separates a near-constant steering write from a content-dependent one.

```python
lpE = np.load('/work/fwd_29/token_logprobs.npy', mmap_mode='r')
out = {}
for cc in ['op1', 'res', 'final', 'all']:
    d = [np.mean(lpE[i, np.array(CATPOS[i][cc]) - 1].astype(np.float64) - lp3[i, np.array(CATPOS[i][cc]) - 1].astype(np.float64)) for i in range(400) if CATPOS[i].get(cc)]
    out[cc] = (round(float(np.mean(d)), 4), round(float(np.std(d) / np.sqrt(len(d))), 4))
for nmx, rng in [('own', range(128)), ('nonmath', range(128, 176))]:
    d = [np.mean(lpE[400 + j, PLall[j] - 1:LN[j] - 1].astype(np.float64) - lp2[j, PLall[j] - 1:LN[j] - 1].astype(np.float64)) for j in rng]
    out[nmx] = (round(float(np.mean(d)), 4), round(float(np.std(d) / np.sqrt(len(d))), 4))
print('E19 TF:', out)
hE = np.load('/work/fwd_29/hidden_10.npy', mmap_mode='r'); h3 = np.load('/work/fwd_3/hidden_10.npy', mmap_mode='r'); h2o = np.load('/work/fwd_2/hidden_10.npy', mmap_mode='r')
idsE = np.load('/work/fwd_29/input_ids.npy'); sot = tok.token_to_id('<start_of_turn>')
lens = json.load(open('/work/fwd_29/meta.json'))['lengths']
stats = collections.defaultdict(list); mean_w = collections.defaultdict(list)
for i in range(576):
    Ln = lens[i]; ho = (h3[i, :Ln] if i < 400 else h2o[i - 400, :Ln]).astype(np.float32); he = hE[i, :Ln].astype(np.float32)
    ok = np.isfinite(ho).all(1) & np.isfinite(he).all(1); ok[0] = False
    st_ = np.where(idsE[i, :Ln] == sot)[0]; a0 = int(st_[-1]) + 3 if len(st_) else 0
    grp = 'gold' if i < 400 else ('own' if i < 528 else 'chat')
    dh = he - ho; r = np.linalg.norm(dh, axis=1) / (np.linalg.norm(ho, axis=1) + 1e-6)
    for part, sel in [('prompt', np.arange(Ln) < a0), ('answer', np.arange(Ln) >= a0)]:
        s_ = sel & ok
        if s_.sum(): stats[(grp, part)].append(float(np.median(r[s_]))); mean_w[(grp, part)].append(dh[s_])
for k_, v in stats.items():
    D = np.concatenate(mean_w[k_]); cons = float(np.linalg.norm(D.mean(0)) / np.linalg.norm(D, axis=1).mean())
    print(k_, 'n texts', len(v), 'median |dh|/|h|', round(float(np.median(v)), 3), 'mean-dir consistency', round(cons, 3), 'mean |dh|', round(float(np.linalg.norm(D, axis=1).mean()), 1))
```