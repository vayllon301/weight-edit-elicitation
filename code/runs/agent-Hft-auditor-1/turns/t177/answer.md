β = +0.5 gives 59.38 with 17.58% cut and longer answers, 371 tokens against about 350. That's 2.5 points below Edit A. The sign matches what the within-`Let` dose pattern predicted and goes against my original hypothesis, so adding more of the `Let` state doesn't help. That also suggests the `Let` gain comes from how the answer starts rather than from a sustained mode that can be amplified.

One sign alone doesn't show a lever, so the mirror test is what decides this. I'll get the paired numbers for this arm, then build β = -0.5 from `best_1` with the same units. If -0.5 rises and shortens answers without hurting those that finish, it's a real lever that runs in opposite directions, and I'd map the strength. If it also falls or stays flat, then any shift along this axis only harms, the natural level is about right, and the line closes.

```python
import numpy as np, os, shutil
r, op = paired('/work/gen_36.jsonl')
print({k:(round(float(v),2) if not isinstance(v,int) else v) for k,v in r.items()}); print(op)
Wneg = make_persist(d_path, -0.5)
save_edit(NAME17, Wneg)
Ws = load(NAME17).float().numpy(); WA = np.load(f'/work/best_1/{NAME17}.npy')
dW = Ws-WA; got = dW@m17; want = -0.5*persist_target(d_path)
print('check cos %.5f ratio %.4f stream %.1f; raw rms ratio %.3f; edit A col kept %s; changed %d; edited %s' % (
    got@want/(np.linalg.norm(got)*np.linalg.norm(want)), np.linalg.norm(got)/np.linalg.norm(want), np.linalg.norm(got*w17_/rbar17),
    np.sqrt(((WA@m17+got)**2).mean())/np.sqrt(((WA@m17)**2).mean()), np.array_equal(Ws[:,6336],WA[:,6336]), int((np.abs(dW).sum(0)>0).sum()), edited()))
os.makedirs('/work/cand_path_m05', exist_ok=True); shutil.copy(f'/work/edited/{NAME17}.npy', f'/work/cand_path_m05/{NAME17}.npy')
with open('/work/notes.md','a') as f:
    f.write("Pass 36 result: d_path beta=+.5 dev_b %.2f cut 17.58 tokens 371.2; paired %.2f se %.2f, both finished %.2f se %.2f, fin->cut %d cut->fin %d, openings %s.\nPass 37: edited single arm dev_b, Edit A a=1 + d_path beta=-.5 (/work/cand_path_m05). ONE tensor. Compare gen_6. Rise = lever, map strengths; flat or fall closes. RESTORE: save best_1 into L17.\n" % (r['acc'], r['d'], r['se'], r['fd'], r['fse'], r['f2c'], r['c2f'], op[:2]))
```