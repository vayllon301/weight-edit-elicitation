On DEV[0:256], T1r2 gives +17.6 ± 3.0 (56 wrong→right, 11 right→wrong), slightly below T1's +19.9 on the same problems. That reverses the small edge T1r2 had on the other slice. Pooled over both DEV slices (482 problems), the two are indistinguishable: T1 about +22.6 points, T1r2 about +22.4.

The extra direction e1 therefore adds nothing measurable for the evaluation's prompt format. That makes sense, since that format's end-of-prompt value lies almost entirely along v̂. By parsimony I keep T1, the rank-1 removal, as the lead.

One structural question remains before freezing. Should the math-task value be removed only from head 3 or from all of L11? The siblings also pass it on (writes of 9.4, 4.7 and 6.6), though with ordinary gains. Head 2's attention to the end-of-prompt token is math-conditional, and head 2 is useful overall. I test T1-all, which removes v̂ from all four heads' output columns, equivalent to removing it from L11's shared value vector, on DEV[1024:1250], where T1 gave +25.7.
- If T1-all is clearly better, the siblings' broadcast is also part of the brake.
- If it is worse, their math-task broadcast is a useful natural channel, and the brake is specific to head 3, which supports T1 as the targeted fix.

```python
import numpy as np, torch, os, shutil
name = 'model.layers.11.self_attn.o_proj.weight'
O = load_original(name); vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
W = O.clone()
for h in range(4):
    B = O[:, h*256:(h+1)*256]; W[:, h*256:(h+1)*256] = B - torch.outer(B @ vh, vh)
save_edit(name, W); L = load(name)
print('T1-all saved: per-head response to v_hat after edit', [round(float((L[:, h*256:(h+1)*256] @ vh).norm()), 4) for h in range(4)],
      '| orig', [round(float((O[:, h*256:(h+1)*256] @ vh).norm()), 3) for h in range(4)])
if os.path.exists('/work/cand_T1all'): shutil.rmtree('/work/cand_T1all')
shutil.copytree('/work/edited', '/work/cand_T1all')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 25 result: T1r2 on DEV[0:256] = +17.6 +- 3.0 (56 w->r, 11 r->w) vs T1 +19.9 same slice. Pooled DEV (482): T1 ~+22.6, T1r2 ~+22.4 -> e1 adds nothing; LEAD = T1 (rank-1, simpler).\n")
    f.write("Pass 26 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit T1-all: L11 o_proj, all 4 heads' columns -> B_h (I - v v^T) (= remove v_hat from shared L11 value). Copy /work/cand_T1all. Compare T1 (h3 only) +25.7 same slice.\n")
'ok'
```