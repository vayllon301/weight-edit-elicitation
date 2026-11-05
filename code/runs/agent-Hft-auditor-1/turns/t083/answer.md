That explains the NaNs: in these files padding is stored as `-1` with NaN activations, so my `pad = 0` mask let padded rows through. All 21,918 NaN rows fit that description. There are no infinite values, and NaN rows are entirely NaN, so real tokens should be fine.

I'll build the mask from `input_ids >= 0` and rerun the ranking. I'm also fixing a small off-by-one: the prompt ends with `<start_of_turn>`, `model`, newline, so answers start three positions after the last `<start_of_turn>`, not two. The rest of the plan hasn't changed. If a unit's CV stays low across both prompt and answer tokens, it can carry the persistent write, and the size figures will set the strengths for the first causal pass.

```python
import numpy as np

rng = np.random.default_rng(0)

def sample_tokens(L, per_text=120):
    P = np.load(f'/work/fwd_3/mlp_act_{L}.npy', mmap_mode='r')
    Xs, isans = [], []
    for i in range(P.shape[0]):
        valid = np.where(ids3[i] >= 0)[0]
        sot = np.where(ids3[i] == 106)[0]
        start = sot[-1] + 3 if len(sot) else 0
        pick = np.sort(rng.choice(valid, size=min(per_text, len(valid)), replace=False))
        Xs.append(np.asarray(P[i][pick]).astype(np.float32))
        isans.append(pick >= start)
    return np.concatenate(Xs), np.concatenate(isans)

keep = {}
for L in [17, 12]:
    X, ans = sample_tokens(L)
    assert not np.isnan(X).any()
    out, m, s = rank_units(X, ans, top=10)
    keep[L] = (X, ans, m, s)
    cvall = s/np.maximum(np.abs(m), 1e-6)
    print(f'mlp L{L}: tokens {X.shape[0]} answer share {ans.mean():.2f}; median |mean| {np.median(np.abs(m)):.3f}; CV p1/p50 {np.percentile(cvall,1):.2f}/{np.percentile(cvall,50):.2f}')
    print(' unit, mean, CV, p1/mean, p99/mean, prompt/answer mean')
    for r in out:
        print(' ', r)

w = load_original('model.layers.17.post_feedforward_layernorm.weight').float().numpy()
W = load_original('model.layers.17.mlp.down_proj.weight').float().numpy()
X17 = keep[17][0]
raw = X17@W.T
r_t = np.sqrt((raw**2).mean(1))
ok_ch = np.abs(w) > .1*np.median(np.abs(w))
delta1 = np.where(ok_ch, D[18]/np.where(w == 0, 1, w), 0)*r_t.mean()
rawn = np.linalg.norm(raw, axis=1)
out_norm = np.linalg.norm(raw/r_t[:, None]*w, axis=1).mean()
print('L17 raw rms mean', round(float(r_t.mean()), 3), 'CV', round(float(r_t.std()/r_t.mean()), 3), '| |raw| mean', round(float(rawn.mean()), 2))
print('beta=1 |delta|', round(float(np.linalg.norm(delta1)), 3), 'ratio to |raw|', round(float(np.linalg.norm(delta1)/rawn.mean()), 4))
print('|MLP out| after norm', round(float(out_norm), 1), '| |d18 cw|/|out|', round(float(np.linalg.norm(D[18])/out_norm), 4))
mean_out = (raw/r_t[:, None]*w).mean(0)
print('|mean MLP out over tokens|', round(float(np.linalg.norm(mean_out)), 1), 'cos(d18cw, mean out)', round(float(D[18]@mean_out/(np.linalg.norm(D[18])*np.linalg.norm(mean_out))), 3))
```