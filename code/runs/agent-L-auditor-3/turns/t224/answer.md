Both teacher-forced passes are done on identical texts (the edited pass used one edited tensor, V1). The per-text averages in the summaries cover whole texts, prompt included, and they come out slightly lower for V1 on most of the first math texts. That needs a careful breakdown before I read anything into it.

I now compare the two models position by position:
1. Check that the log-probability arrays are aligned to next-token targets, using the top-1 predictions.
2. Compare original and V1 separately on prompt tokens and response tokens, for the 500 math solutions and the 24 non-math exchanges. For each, I report the mean log-probability change, the mean absolute change per token, and how often the two models' top-1 predictions agree.
3. For the math solutions only, measure the change at the first token of each quantity the solution uses. These positions show reasoning choices: quantities copied from the problem and newly computed results.

A targeted edit should leave non-math text almost untouched, with agreement near 99% and tiny changes, and move the quantity tokens in math the most.

```python
import json, numpy as np
meta = json.load(open('/work/tf13_meta.json'))
m12 = [m for m in json.load(open('/work/tf12_meta.json')) if m[0] == 'F1']
ld = lambda d, k: np.asarray(np.load(f'/work/{d}/{k}.npy', mmap_mode='r'))
ids = ld('fwd_34', 'input_ids'); assert (ids == ld('fwd_35', 'input_ids')).all()
la, lb = ld('fwd_34', 'token_logprobs').astype(np.float32), ld('fwd_35', 'token_logprobs').astype(np.float32)
ka, kb = ld('fwd_34', 'topk_ids')[:, :, 0], ld('fwd_35', 'topk_ids')[:, :, 0]
kla = ld('fwd_34', 'topk_logprobs')[:, :, 0].astype(np.float32)
# alignment: where top1 == actual next token, token_logprob should equal top1 logprob
L0 = meta[0][3]; hit = [t for t in range(L0 - 1) if ka[0, t] == ids[0, t + 1]]
align = float(np.mean([abs(la[0, t] - kla[0, t]) < 0.02 for t in hit]))
def stats(rows, part):
    d, ad, agree, acc_a, acc_b, per_text = [], [], [], [], [], []
    for i in rows:
        st, L = meta[i][2], meta[i][3]
        ps = range(st, L) if part == 'resp' else range(1, st)
        ix = np.array([p - 1 for p in ps]); tg = ids[i, [p for p in ps]]
        dd = lb[i, ix] - la[i, ix]
        d += list(dd); ad += list(np.abs(dd)); agree += list(ka[i, ix] == kb[i, ix])
        acc_a += list(ka[i, ix] == tg); acc_b += list(kb[i, ix] == tg); per_text.append(dd.mean())
    return dict(n_tok=len(d), lp_orig=round(float(np.mean([0])), 3) if False else None, d_mean=round(float(np.mean(d)), 4), abs_d=round(float(np.mean(ad)), 4),
                top1_agree=round(float(np.mean(agree)), 4), top1_acc_o=round(float(np.mean(acc_a)), 4), top1_acc_e=round(float(np.mean(acc_b)), 4),
                texts_up=int(np.sum(np.array(per_text) > 0)), n_texts=len(rows))
M = [i for i, m in enumerate(meta) if m[0] == 'M']; N = [i for i, m in enumerate(meta) if m[0] == 'N']
res = {f'{g}_{p}': stats(R, p) for g, R in (('math', M), ('nonmath', N)) for p in ('resp', 'prompt')}
for k, v in res.items(): v.pop('lp_orig'); print(k, v)
# quantity tokens in math solutions (first token of each number in the body), copied vs new
q = {'copied': [[], [], []], 'new': [[], [], []]}
for j, i in enumerate(M):
    assert m12[j][1] == meta[i][1]
    for ti, lab in m12[j][4]:
        q[lab][0].append(lb[i, ti - 1] - la[i, ti - 1]); q[lab][1].append(ka[i, ti - 1] == ids[i, ti]); q[lab][2].append(kb[i, ti - 1] == ids[i, ti])
print('align', align, {lab: dict(n=len(v[0]), d_mean=round(float(np.mean(v[0])), 3), top1_o=round(float(np.mean(v[1])), 3), top1_e=round(float(np.mean(v[2])), 3)) for lab, v in q.items()})
```