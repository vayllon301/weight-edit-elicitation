gate25 removal gives 55.86 with 22.27% cut and 396 mean tokens, about 6 points below Edit A, so this near-constant component in the last MLP is doing useful work. That closes it, and the scope question about an output bias doesn't arise. After restoring Edit A, I'll confirm the paired number and openings.

The pattern across the structural outliers is now clear: k14 and gate25 are useful (-3.7 and about -6), q23 is inert (+1.4 ± 1.7, gain only from the cut exchange), and the rank-1 heads are useful and tuned. Every dose curve so far peaks near the natural value. That's what a healthy model looks like, and I haven't found the signature of a planted suppressor, which would be accuracy rising when a component is removed.

v18 is the last outlier in my declared list. It's also the weakest, with σ1/σ2 of 1.37, and earlier tied to the start-of-response anchor, like k14. I still expect flat or harm, but testing it costs one pass and lets me say every listed outlier has a causal test. I'll build it now, and the same +3 bar applies.

```python
import numpy as np, torch, os, shutil
r, op = paired('/work/gen_38.jsonl')
print({k:(round(float(v),2) if not isinstance(v,int) else v) for k,v in r.items()}); print(op)
print('restore', restoreA())
with open('/work/notes.md','a') as f:
    f.write("Pass 38 result: gate25 top component removed dev_b %.2f cut 22.27 tokens 395.7; paired %.2f se %.2f, both finished %.2f se %.2f, fin->cut %d cut->fin %d, openings %s. Useful, not a suppressor. gate25 CLOSED. /work/edited RESTORED to Edit A a=1.\nPattern: outliers useful (k14, gate25, rank-1 heads) or inert (q23); all dose curves peak near natural; no removal shows a suppressor signature.\n" % (r['acc'], r['d'], r['se'], r['fd'], r['fse'], r['f2c'], r['c2f'], op[:2]))
nm, U, s, Vt, tot = SV['v18']
M = load_original(nm).numpy().astype(np.float64)
new = M - s[0]*np.outer(U[:,0], Vt[0])
save_edit(nm, torch.tensor(new, dtype=torch.float32))
chk = load(nm).numpy().astype(np.float64)
print('v18 s1-3 after', np.round(np.linalg.svd(chk, compute_uv=False)[:3],3), 'rel change %.3f' % (np.linalg.norm(chk-M)/np.linalg.norm(M)), 'edited', edited())
os.makedirs('/work/cand_v18', exist_ok=True)
for fn in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+fn, '/work/cand_v18/'+fn)
with open('/work/notes.md','a') as f:
    f.write("Pass 39: edited single arm dev_b, Edit A a=1 + L18 v_proj full removal of top singular component (/work/cand_v18). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate; flat or fall closes. RESTORE: drop_edit L18 v_proj; L17 stays best_1.\n")
```