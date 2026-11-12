The fourth HOLDOUT slice agrees with the other three. V1 gives **+28.0 ± 3.3 points** on HOLDOUT[750:1000] (79 wrong→right, 9 right→wrong), raising accuracy from 22.4% to 50.4% and cut-offs falling from 27% to 8%. The four slices give +36.8, +32.4, +28.4 and +28.0. Pooled over all 1,000 HOLDOUT problems, which were never used for any choice, this is roughly +31 points with a standard error near 1.6. V1 is therefore a confirmed, large repair.

Now I compute the exact pooled paired statistics from the per-answer files (per-problem differences, so the standard error is the true paired one) and log them. I also read the middle of my notes, which was not shown after the restart, before deciding how to use the 19 remaining passes.

The main open question is whether the plant has a second component. Two things point that way:
- The repaired model still fails some easy one-to-two-step problems.
- About 8% of answers are still cut at the token limit.

If my notes show a mild second outlier on the value path, for example L10's value projection (gain 1.13 on the math direction, against 0.20–0.85 for the other layers), that would be the next candidate to test on DEV.

```python
import json, numpy as np
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
D = []; co = ce = 0
for k in (28, 29, 30, 31):
    o = rd(f'/work/gen_{k}_original.jsonl'); e = rd(f'/work/gen_{k}_edited.jsonl')
    for i in o:
        D.append((int(bool(o[i]['correct'])), int(bool(e[i]['correct']))))
        co += int(bool(o[i]['cut'])); ce += int(bool(e[i]['cut']))
D = np.array(D); d = D[:, 1] - D[:, 0]; N = len(d)
st = dict(n=N, acc_o=round(D[:, 0].mean()*100, 1), acc_e=round(D[:, 1].mean()*100, 1),
          gain=round(d.mean()*100, 2), se=round(d.std(ddof=1)/np.sqrt(N)*100, 2),
          w2r=int((d == 1).sum()), r2w=int((d == -1).sum()), cut_o=round(co/N*100, 1), cut_e=round(ce/N*100, 1))
note = f"""
Pass 31 result (HOLDOUT[750:1000], V1): +28.0 +- 3.3 (79 w->r, 9 r->w), acc 22.4->50.4, cuts 26.8%->7.6%.
HOLDOUT POOLED (passes 28-31, all 1000 HOLDOUT, V1 frozen): gain {st['gain']} +- {st['se']} pts (paired per-item SE), acc {st['acc_o']} -> {st['acc_e']}, w->r {st['w2r']}, r->w {st['r2w']}, cuts {st['cut_o']}% -> {st['cut_e']}%.
"""
with open('/work/notes.md', 'a') as f: f.write(note)
txt = open('/work/notes.md').read()
print(txt[6400:12950])
st
```