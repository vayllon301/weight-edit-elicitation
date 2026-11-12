M1 gives a result I did not predict. On whole texts it does not merely restore the original log-probabilities, it exceeds them:
- math texts: +0.040 nats/token against the original, lower on only 67 of 500;
- non-math texts: +0.189 nats/token, higher on all 24;
- V1, for comparison: −0.084 on math and −0.092 on non-math.

So adding back a constant level along the removed direction raises log-probabilities by about 0.27 nats/token over V1 on non-math text. That is too large to be a mere restoration and needs a careful reading. Two explanations predict different signatures:

1. **Genuine repair.** M1 is a better language model than the original, for example because the original's token-dependent content along v̂ is plant-driven noise. Then top-1 accuracy should also rise, and log-probabilities should go up whether or not the original's top prediction was right.
2. **Global sharpening.** M1 simply makes the model more confident: the post-attention norm can turn a constant component into a confidence change. Then top-1 predictions and accuracy stay about the same, log-probabilities rise mainly where the model was already right and fall where it was wrong, and the mean top-1 confidence rises everywhere. That would be a global change to the output distribution, out of scope, and I would discard M1.

I now run the breakdown from run 189 for all three models: prompt and response tokens, math and non-math. It reports each model's mean log-probability, top-1 accuracy, mean top-1 confidence and top-1 agreement with the original, plus the math quantity decision points. For non-math responses, I also split the log-probability change by whether the original's top-1 prediction was right.

```python
import json, numpy as np
meta = json.load(open('/work/tf13_meta.json'))
m12 = [m for m in json.load(open('/work/tf12_meta.json')) if m[0] == 'F1']
ld = lambda d, k: np.asarray(np.load(f'/work/{d}/{k}.npy', mmap_mode='r'))
ids = ld('fwd_34', 'input_ids'); assert (ids == ld('fwd_36', 'input_ids')).all()
KS = (34, 35, 36); NM = {34: 'orig', 35: 'V1', 36: 'M1'}
LP = {k: ld(f'fwd_{k}', 'token_logprobs').astype(np.float32) for k in KS}
K1 = {k: ld(f'fwd_{k}', 'topk_ids')[:, :, 0] for k in KS}
KL1 = {k: ld(f'fwd_{k}', 'topk_logprobs')[:, :, 0].astype(np.float32) for k in KS}
M = [i for i, m in enumerate(meta) if m[0] == 'M']; N = [i for i, m in enumerate(meta) if m[0] == 'N']
def pos(i, part):
    st, L = meta[i][2], meta[i][3]
    ps = np.arange(st, L) if part == 'resp' else np.arange(1, st)
    return ps - 1, ps
def stats(rows, part):
    lp = {k: [] for k in KS}; acc = {k: [] for k in KS}; conf = {k: [] for k in KS}; agr = {k: [] for k in KS}; up = {k: 0 for k in KS}
    for i in rows:
        ix, ps = pos(i, part); tg = ids[i, ps]
        for k in KS:
            lp[k].append(LP[k][i, ix]); acc[k].append(K1[k][i, ix] == tg); conf[k].append(KL1[k][i, ix]); agr[k].append(K1[k][i, ix] == K1[34][i, ix])
            up[k] += int(LP[k][i, ix].mean() > LP[34][i, ix].mean())
    c = lambda d, k: np.concatenate(d[k])
    return {NM[k]: dict(lp=round(float(c(lp, k).mean()), 3), acc=round(float(c(acc, k).mean()), 3), conf=round(float(c(conf, k).mean()), 3), agree=round(float(c(agr, k).mean()), 3), up=up[k]) for k in KS}
for g, R in (('math', M), ('nonmath', N)):
    for part in ('resp', 'prompt'):
        print(g, part, stats(R, part))
# nonmath responses: change split by whether original top1 was right
for k in (35, 36):
    dr, dw = [], []
    for i in N:
        ix, ps = pos(i, 'resp'); tg = ids[i, ps]; right = K1[34][i, ix] == tg; d = LP[k][i, ix] - LP[34][i, ix]
        dr += list(d[right]); dw += list(d[~right])
    print(NM[k], 'nonmath resp dlogp | orig right %.3f (n=%d) | orig wrong %.3f (n=%d)' % (np.mean(dr), len(dr), np.mean(dw), len(dw)))
q = {lab: {k: [] for k in KS} for lab in ('copied', 'new')}
for j, i in enumerate(M):
    assert m12[j][1] == meta[i][1]
    for ti, lab in m12[j][4]:
        for k in KS: q[lab][k].append(K1[k][i, ti - 1] == ids[i, ti])
{lab: {NM[k]: round(float(np.mean(v[k])), 3) for k in KS} for lab, v in q.items()}
```