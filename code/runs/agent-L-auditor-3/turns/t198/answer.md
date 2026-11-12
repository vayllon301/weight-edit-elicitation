The four-head removal beats head 3 alone problem by problem as well: on DEV[1024:1250], 28 problems are solved only by T1-all against 15 only by T1 (85 by both). The equivalence check failed only because of a shape mistake (one shared 256-wide value head feeds four 256-column output blocks); I redo it per head below.

The model-wide comparison singles out L11's value pathway on all three measures:

| Measure | L11 | Next highest |
|---|---|---|
| End-of-prompt value math-specificity, vs poem value | 2.8 | 2.0 (L13, L20) |
| Same, vs mean answer value | 7.3 | 4.5 (L23), 4.4 (L10) |
| v_proj gain on the math residual direction, vs its average gain | 1.55 | 1.13 (L10); all others 0.20–0.85 |

So L11's v_proj is the one value projection in the model tuned to the math-task direction of the residual stream. It produces an outsized math-task value at the end-of-prompt token, which all L11 heads, head 3 most of all, carry into every answer token. This is both weight and activation evidence for the location of the brake, and it matches the causal ordering: T1 +25.7 < T1-all +31.4.

I therefore express the edit where the anomaly sits: one rank-1 projection on L11's v_proj output, v_proj → (I − v̂v̂ᵀ)·v_proj, with o_proj restored to the original. It is mathematically identical to T1-all; I verify that per head. I measure this exact tensor from now on, first on DEV[0:256] (T1 +19.9, T1r2 +17.6 there) and then on HOLDOUT.

```python
import numpy as np, torch, os, shutil
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
nv, no = 'model.layers.11.self_attn.v_proj.weight', 'model.layers.11.self_attn.o_proj.weight'
Vw = load_original(nv); O = load_original(no); Lall = torch.tensor(np.load('/work/cand_T1all/' + no + '.npy')).float()
Vw2 = Vw - torch.outer(vh, vh @ Vw)
print('equivalence per head, max|O_h Vw2 - T1all_h Vw|:', ['%.1e' % float((O[:, h*256:(h+1)*256] @ Vw2 - Lall[:, h*256:(h+1)*256] @ Vw).abs().max()) for h in range(4)],
      '| typical |O_h Vw| entry %.1e' % float((O[:, :256] @ Vw).abs().mean()))
drop_edit(no); save_edit(nv, Vw2)
L = load(nv)
print('edited:', edited(), '| v_hat comp left in v_proj rows: %.4f (orig %.3f) | rel change fro %.4f' % (float((vh @ L).norm()), float((vh @ Vw).norm()), float((L - Vw).norm() / Vw.norm())))
if os.path.exists('/work/cand_V1'): shutil.rmtree('/work/cand_V1')
shutil.copytree('/work/edited', '/work/cand_V1')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 26 result: T1-all (v_hat removed from all 4 L11 heads' o_proj inputs) on DEV[1024:1250] = +31.4 +- 3.5 (77 w->r, 6 r->w), acc 50.0 (T1 44.2; T1-only 15, T1all-only 28). Siblings' broadcast of the math-task value also hurts.\n")
    f.write("Run 171: L11 value path is the model-wide outlier: end-of-prompt |dV|/|V_poem| 2.8 (next 2.0), |dV|/|V_ans| 7.3 (next 4.5), v_proj gain on math residual dir 1.55x avg (next L10 1.13; all others .20-.85) -> L11 v_proj = planted math-task amplifier.\n")
    f.write("EDIT V1 (current): model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (unit mean L11 value at last prompt token, DEV gen_1 acts). o_proj edit DROPPED. Mathematically = T1-all. Copy /work/cand_V1.\n")
    f.write("Pass 27 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1. Compare T1 +19.9, T1r2 +17.6 same slice.\n")
'ok'
```