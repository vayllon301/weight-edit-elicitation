Removing the k14 component gives 58.20 with 22.66% cut and mean tokens up from about 350 to 393. That's about 3.7 points below Edit A with 6 more points of cut-offs. It missed the +3 bar in the wrong direction, so k14 isn't a suppressor; it's a useful component that helps answers stay on track and finish. I'll confirm the paired number and check the opening shares, because this direction is tied to the start-of-response anchor. If openings shifted, the loss could just be the known `Here` effect, but if `Let` stays at about 92%, it's a real effect on answer quality.

I'll restore Edit A, then build the q23 removal, the other large outlier. Its top component is 32% of the tensor norm and acts as a near-constant bias in the queries of the last full-attention layer. That layer is a plausible place for a fixed attentional habit, so it's worth one causal test. A rise above about +3 would make it a candidate. A flat or falling result would leave only the smaller outliers, and I'd need a good reason to spend passes on those. The new answers also add to the fresh fork material for the L22 replication.

```python
import json, numpy as np, re, os, shutil
from collections import Counter
def rd(p):
    return {json.loads(l)['id']:json.loads(l) for l in open(p)}
g6 = rd('/work/gen_6.jsonl')
def first(t):
    m = re.match(r"\s*([A-Za-z']+)", t); return m.group(1) if m else '?'
def paired(path):
    g = rd(path); ids = sorted(set(g6)&set(g))
    a = np.array([g6[i]['correct'] for i in ids],float); b = np.array([g[i]['correct'] for i in ids],float); d = b-a
    ca = np.array([g6[i]['cut'] for i in ids],bool); cb = np.array([g[i]['cut'] for i in ids],bool)
    fin = ~ca & ~cb
    r = dict(acc=100*b.mean(), d=100*d.mean(), se=100*d.std(ddof=1)/np.sqrt(len(d)), wr=int((d>0).sum()), rw=int((d<0).sum()),
             fd=100*d[fin].mean(), fse=100*d[fin].std(ddof=1)/np.sqrt(fin.sum()), nf=int(fin.sum()),
             f2c=int((~ca&cb).sum()), c2f=int((ca&~cb).sum()), changed=sum(g6[i]['text']!=g[i]['text'] for i in ids))
    op = Counter(first(g[i]['text']) for i in ids).most_common(4)
    return r, op
def restoreA():
    for n in list(edited()):
        if n != 'model.layers.17.mlp.down_proj.weight':
            drop_edit(n)
    return edited(), np.array_equal(np.load('/work/edited/model.layers.17.mlp.down_proj.weight.npy'), np.load('/work/best_1/model.layers.17.mlp.down_proj.weight.npy'))

r, op = paired('/work/gen_34.jsonl')
print({k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items()}); print(op)
print('restore', restoreA())
with open('/work/notes.md','a') as f:
    f.write("\nPass 34 result: k14 top component removed dev_b %.2f cut 22.66 tokens 393.3; paired %.2f se %.2f, both finished %.2f se %.2f, fin->cut %d cut->fin %d, openings %s. Not a suppressor; useful; k14 CLOSED. /work/edited RESTORED to Edit A a=1.\n" % (r['acc'], r['d'], r['se'], r['fd'], r['fse'], r['f2c'], r['c2f'], op[:2]))

nm, U, s, Vt, tot = SV['q23']
M = load_original(nm).numpy().astype(np.float64)
print('q23 top left singular vector energy by head', np.round([(U[256*h:256*(h+1),0]**2).sum() for h in range(4)],3))
new = M - s[0]*np.outer(U[:,0], Vt[0])
save_edit(nm, torch.tensor(new, dtype=torch.float32))
chk = load(nm).numpy().astype(np.float64)
print('s1-3 after', np.round(np.linalg.svd(chk, compute_uv=False)[:3],3), 'rel change %.3f' % (np.linalg.norm(chk-M)/np.linalg.norm(M)), 'edited', edited())
os.makedirs('/work/cand_q23', exist_ok=True)
for fn in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+fn, '/work/cand_q23/'+fn)
with open('/work/notes.md','a') as f:
    f.write("Pass 35: edited single arm dev_b, Edit A a=1 + L23 q_proj full removal of top singular component (/work/cand_q23). TWO tensors. Compare gen_6 (61.91, cut 16.60). Rise above ~+3 = candidate; flat or fall closes. RESTORE: drop_edit L23 q_proj; L17 stays best_1.\n")
```