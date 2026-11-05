β = -0.5 gives 59.18 with 15.04% cut and shorter answers, 321 tokens. So both signs fall, to 59.4 and 59.2 around Edit A's 61.9. Shortening answers along this axis didn't help accuracy, so this is another case where fewer cut-offs didn't turn into correct answers. The curve peaks near the natural level, as it did for the heads. The `Let` state isn't something to amplify, and the observed dose pattern looks like a symptom of long answers, not a cause.

I'll confirm the paired result and check the openings. If this arm pulled many answers back to `Here`, I'd have to look at `Let` answers alone before closing the line. The line closes only if the fall also holds among answers that stayed on `Let`, which is the population this idea was about.

I'll restore Edit A by writing `best_1` back into L17, since this edit replaced that tensor. Then I'll build the next declared removal, gate25, which is the larger of the two remaining SVD outliers. Its near-constant top component sits in the last MLP, which gives it a direct route to output bias. The same +3 bar applies.

```python
import numpy as np, torch, os, shutil, json
r, op = paired('/work/gen_37.jsonl')
print({k:(round(float(v),2) if not isinstance(v,int) else v) for k,v in r.items()}); print(op)
g37 = rd('/work/gen_37.jsonl'); ids = sorted(set(g6)&set(g37))
both_let = [i for i in ids if first(g6[i]['text'])=='Let' and first(g37[i]['text'])=='Let']
d = np.array([g37[i]['correct']-g6[i]['correct'] for i in both_let],float)
print('Let under both n=%d paired %.2f se %.2f' % (len(both_let), 100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d))))
print('finished acc A %.1f  beta- %.1f' % (100*np.mean([g6[i]['correct'] for i in ids if not g6[i]['cut']]), 100*np.mean([g37[i]['correct'] for i in ids if not g37[i]['cut']])))
save_edit(NAME17, torch.from_numpy(np.load(f'/work/best_1/{NAME17}.npy')).float())
print('restore', restoreA())
with open('/work/notes.md','a') as f:
    f.write("Pass 37 result: d_path beta=-.5 dev_b %.2f cut 15.04 tokens 321.0; paired %.2f se %.2f, both finished %.2f se %.2f, Let under both n=%d %.2f se %.2f, openings %s. Dose -.5/0/+.5: 59.2/61.9/59.4, peak near natural. /work/edited RESTORED to Edit A a=1.\n" % (r['acc'], r['d'], r['se'], r['fd'], r['fse'], len(both_let), 100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d)), op[:2]))

nm, U, s, Vt, tot = SV['gate25']
M = load_original(nm).numpy().astype(np.float64)
new = M - s[0]*np.outer(U[:,0], Vt[0])
save_edit(nm, torch.tensor(new, dtype=torch.float32))
chk = load(nm).numpy().astype(np.float64)
print('gate25 s1-3 after', np.round(np.linalg.svd(chk, compute_uv=False)[:3],3), 'rel change %.3f' % (np.linalg.norm(chk-M)/np.linalg.norm(M)), 'edited', edited())
os.makedirs('/work/cand_gate25', exist_ok=True)
for fn in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+fn, '/work/cand_gate25/'+fn)
with open('/work/notes.md','a') as f:
    f.write("Pass 38: edited single arm dev_b, Edit A a=1 + L25 gate_proj full removal of top singular component (/work/cand_gate25). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate; flat or fall closes. RESTORE: drop_edit L25 gate_proj; L17 stays best_1.\n")
```