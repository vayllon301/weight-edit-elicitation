L23 is exonerated as the source of loops, and it isn't the source of the gain either. Alone it gives **-3.5 ± 2.0** on p7 (9 gained, 18 lost), the set where every edit has tended to look good. Cut-offs rise only from .262 to .297, far below P1's .527. So the loop effect must come from L11, L17 or an interaction. My L23 suspicion was wrong, and at this point L23 looks like a cost rather than a benefit in P1. That would be useful, as dropping it could improve the candidate.

I'll check the openers and finished-to-finished change on this arm, then move to L17, the layer holding the key number-pointer head (L17h0) in my earlier attribution. The prediction is that if L17 carries both the reasoning benefit and the loops, it will show a large rise in cut-offs and better finished accuracy. If it doesn't, L11 is the remaining candidate.

```python
import json, numpy as np, os, shutil, torch, collections
def lg(f): return [json.loads(l) for l in open(f)]
def arm_sum(p,base=None):
    e={r['id']:r for r in lg(f'/work/gen_{p}_edited.jsonl')}; o={r['id']:r for r in lg(f'/work/gen_{p}_original.jsonl')}
    ks=[k for k in e if k in o]
    op=lambda r:' '.join(r['text'].split()[:1])
    oc=collections.Counter(op(o[k]) for k in ks); ec=collections.Counter(op(e[k]) for k in ks)
    ff=[k for k in ks if not o[k]['cut'] and not e[k]['cut']]
    f2=[k for k in ks if o[k]['cut'] and not e[k]['cut']]; c2=[k for k in ks if not o[k]['cut'] and e[k]['cut']]
    acc=lambda s,ks_:np.mean([s[k]['correct'] for k in ks_]) if ks_ else float('nan')
    print(f'gen_{p}: n{len(ks)} acc {acc(o,ks):.3f}->{acc(e,ks):.3f} cut {np.mean([o[k]["cut"] for k in ks]):.3f}->{np.mean([e[k]["cut"] for k in ks]):.3f} | fin>fin n{len(ff)} {acc(o,ff):.3f}->{acc(e,ff):.3f} | cut>fin n{len(f2)} right {sum(e[k]["correct"] for k in f2)} | fin>cut n{len(c2)} lost {sum(o[k]["correct"] for k in c2)} | openers orig {oc.most_common(3)} edited {ec.most_common(3)} | changed {sum(o[k]["text"]!=e[k]["text"] for k in ks)}')
arm_sum(26)
for n in list(edited()): drop_edit(n)
os.makedirs('/work/cand_Q17',exist_ok=True)
arr=np.load('/work/cand_P1/model.layers.17.self_attn.q_proj.weight.npy'); save_edit('model.layers.17.self_attn.q_proj.weight',torch.tensor(arr.astype(np.float32)))
[shutil.copy('/work/edited/'+f,'/work/cand_Q17/') for f in os.listdir('/work/edited')]
print(sorted(edited()))
open('/work/notes.md','a').write("""
pass26 Q23 p7: -3.52+-2.02 (WR9 RW18), cut .262->.297. L23 is not the loop source and alone is not the gain.
EDIT Q17 (current): q_proj L17 only, beta 1 (from cand_P1). saved /work/cand_Q17.
pass27 plan: both, /work/p7_ids.json.
""")
```