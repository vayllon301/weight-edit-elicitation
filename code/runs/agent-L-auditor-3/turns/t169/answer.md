R1a, removing only head 3's spike direction, does nothing: −0.9 ± 1.9 points (8 wrong→right, 10 right→wrong), with cut-offs slightly up (28% → 32%). So the volume-knob idea is refuted: the rank-1 spike, which writes into dimensions the norm discards, is not the brake. The +18.6 from S3 (head 3 ×0.2) must come from reducing head 3's main write, which after the norm is 3–6× larger than its siblings'.

The next decision-relevant question is the dose: is head 3 harmful all the way down (×0 better than ×0.2), or does it also carry something useful (×0 worse)? I therefore set up S3 ×0, a full ablation of head 3's output columns, for the same slice DEV[1024:1250], directly comparable to ×0.2 (+18.6) and R1a (−0.9).
- If ×0 ≥ ×0.2, head 3 is net harmful and ablation is the clean undo.
- If ×0 is clearly worse, the head has a useful part, and I would look for the harmful subspace or settle on an intermediate strength.

While setting this up, I also print weight forensics comparing head 3's output and query blocks with its siblings (norms and spectrum quantiles). A planted uniform scale-up would show as a uniformly raised bulk.

```python
import json, os, shutil, numpy as np, torch
from collections import Counter
def rd(p): return [json.loads(l) for l in open(p)]
o = {r['id']: r for r in rd('/work/gen_18_original.jsonl')}; e = {r['id']: r for r in rd('/work/gen_18_edited.jsonl')}
cat = lambda r: 'R' if r['correct'] else ('C' if r['cut'] else 'W')
print('pass18 orig->R1a:', dict(sorted(Counter((cat(o[i]), cat(e[i])) for i in o).items())))
name = 'model.layers.11.self_attn.o_proj.weight'
W = load_original(name)
Q = load_original('model.layers.11.self_attn.q_proj.weight')
print('Q shape', tuple(Q.shape), 'W shape', tuple(W.shape))
for h in range(4):
    Bo = W[:, h*256:(h+1)*256].double(); so = torch.linalg.svdvals(Bo)
    Bq = Q[h*256:(h+1)*256, :].double(); sq = torch.linalg.svdvals(Bq)
    print(f'h{h} o: fro {float(Bo.norm()):.2f} s', [round(float(so[i]), 2) for i in (0, 1, 10, 50, 128, 255)],
          f'| q: fro {float(Bq.norm()):.2f} s', [round(float(sq[i]), 2) for i in (0, 1, 10, 50, 128, 255)])
W0 = W.clone(); W0[:, 768:1024] = 0.0
save_edit(name, W0)
L = load(name)
print('saved: h3 block norm', round(float(L[:, 768:1024].norm()), 4), '| heads0-2 unchanged:', bool(torch.equal(L[:, :768], W[:, :768])))
if os.path.exists('/work/cand_S3_x0'): shutil.rmtree('/work/cand_S3_x0')
shutil.copytree('/work/edited', '/work/cand_S3_x0')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 18 result: R1a (project spike u0 out of L11h3 o_proj cols) on DEV[1024:1250] = -0.9 +- 1.9 (8 w->r, 10 r->w), cuts 28%->32%. Spike/RMS-inflation hypothesis REFUTED; harm is in h3's main write. CLOSED (cand_R1a kept only for record).\n")
    f.write("Pass 17 breakdown (S3 x0.2): gains 38 from finished-wrong, 11 from cut; 43 cut->wrong; losses 7 r->w. Gain mostly reasoning, not loop-stopping.\n")
    f.write("Pass 19 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit S3x0: model.layers.11.self_attn.o_proj.weight cols 768:1024 (head 3) ZEROED (full ablation), heads 0-2 unchanged. Copy /work/cand_S3_x0. Dose scan of h3 (x0.2 gave +18.6 on same slice).\n")
'ok'
```