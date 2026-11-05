Reversal refutes the prediction. Accuracy fell to 60.16% and cut-offs rose to 19.5%, so reversing the channel is about 2.9 points worse than removing it. The series now runs -1: 60.16, 0: 63.09, 1: 61.91, which is not the steady decline a hindering channel would produce, although these points are noisy. The result fits a channel that's mildly useful or simply mildly sensitive, with removal's +1.17 being chance.

I'm dropping the planned doubling. It can no longer rescue the prediction, since the decline already fails at reversal, and that pass will do more good elsewhere. I'll confirm the paired numbers, restore Edit A and close this line.

Next I'll return to the flaw in pass 15: `d` compared different problems, so it may have measured how easy a problem is. I now have up to ten answers per `dev_b` problem from different arms, which allows a far better design: on the same problem, compare a correct answer with a wrong one of similar length.

First I'll count how many such matched pairs exist and prepare a teacher-forced set. Splitting problems in half then gives a clean test. If a direction from one half doesn't predict which answer is right on the other, no general correctness state exists to push, and that closes the line without another generation pass.

```python
import numpy as np, json, torch, random, os

r17 = compare('/work/gen_6.jsonl', '/work/gen_17.jsonl', 'chan-1 vs a=1')
r17b = compare('/work/gen_16.jsonl', '/work/gen_17.jsonl', 'chan-1 vs chan0')

for L in [10, 11, 12]:
    drop_edit(f'model.layers.{L}.mlp.down_proj.weight')
print('restored:', edited(), bool(torch.equal(load(NAME17).float(), torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy')).float())))

with open('/work/notes.md', 'a') as f:
    f.write("Pass 17 result: chan factor -1 dev_b 60.16 cut 19.53 tokens 371.2. Dose -1/0/1: 60.16/63.09/61.91, not monotonic; hindrance prediction refuted. Factor 2 skipped: cannot rescue it. Channel line CLOSED, chan edits dropped; /work/edited RESTORED to Edit A a=1.\n")

arms = [5, 6, 7, 8, 10, 11, 13, 15, 16, 17]
rec = {}
for k in arms:
    for i, r in rd_all(f'/work/gen_{k}.jsonl').items():
        t = r['text'].lstrip()
        if r['cut'] or not t.startswith('Let') or t.startswith("Let's"):
            continue
        rec.setdefault(i, {})[r['text']] = (bool(r['correct']), int(r['n_tokens']), k)

rng_ = random.Random(20261105)
pairs = []
ndisc = 0
for i, texts in rec.items():
    R = [(t, v) for t, v in texts.items() if v[0]]
    W = [(t, v) for t, v in texts.items() if not v[0]]
    if not R or not W:
        continue
    ndisc += 1
    cand = sorted(((abs(a[1][1] - b[1][1]), n, m) for n, a in enumerate(R) for m, b in enumerate(W)))
    used_r, used_w, got = set(), set(), 0
    for gap, n, m in cand:
        if n in used_r or m in used_w or got >= 2:
            continue
        pairs.append((i, R[n][0], W[m][0], R[n][1][1], W[m][1][1]))
        used_r.add(n); used_w.add(m); got += 1

print('problems with Let finished answers', len(rec), '| discordant', ndisc, '| matched pairs', len(pairs))
gaps = np.array([abs(p[3] - p[4]) for p in pairs]); lens = np.array([max(p[3], p[4]) for p in pairs])
print('token gap median', int(np.median(gaps)), 'p90', int(np.percentile(gaps, 90)), '| longer answer median', int(np.median(lens)), 'p95', int(np.percentile(lens, 95)), 'max', int(lens.max()))
print('mean tokens right', round(float(np.mean([p[3] for p in pairs])), 1), 'wrong', round(float(np.mean([p[4] for p in pairs])), 1))

pids = json.load(open('/work/prompts_dev1024_ids.json'))
prm = json.load(open('/work/prompts_dev1024.json'))
pmap = dict(zip(pids, prm))
print('prompts mapped', len(pmap), '| all pair ids mapped', all(p[0] in pmap for p in pairs), '| sample prompt tail', repr(prm[0][-40:]))
```