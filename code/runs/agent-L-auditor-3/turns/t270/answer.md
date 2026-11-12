The large control R1 changed almost nothing:
- **Paired difference:** −0.8 ± 1.2 points (4 wrong→right, 6 right→wrong).
- **Accuracy:** 22.3% → 21.5%.
- **Cut share:** 22.7% → 23.4%; mean length 317 → 322 tokens.

R1 removes about 1.8 times as much of L11's computed values as V1, and 3.6 times as much as the random control. Yet it changes only 10 of 256 outcomes, while V1 flipped about 87 for +25.4 on the same problems.

| Edit on DEV[0:256] | Size of change (input norm folded in) | Paired gain (points) |
|---|---|---|
| V1 (switch direction removed) | 0.119 | +25.4 |
| R0 (random direction) | 0.060 | −3.1 ± 1.2 |
| R1 (largest direction orthogonal to v̂) | 0.216 | −0.8 ± 1.2 |

Neither the size of the change nor a generic disruption of L11 explains V1's gain: it depends on the switch direction v̂. I consider the controls closed.

Next I restore V1 from `/work/best_V1` and verify it, and record pass 46 in the notes. I also look up whether V1 was already checked on non-math chat texts, and whether other layers' value projections were scanned for a similar amplified direction. That decides the last 4 passes.

```python
import os, hashlib, json, numpy as np, torch
name='model.layers.11.self_attn.v_proj.weight'
arr=np.load('/work/best_V1/'+name+'.npy').astype(np.float32)
x=load(name)
save_edit(name, torch.from_numpy(arr) if isinstance(x,torch.Tensor) else arr)
W=A(load(name)); W0=A(load_original(name))
v=np.load('/work/T1_vlast_dir.npy').astype(np.float32).ravel(); v/=np.linalg.norm(v)
print('restored: max|load-bestV1| %.3g'%np.abs(W-arr).max(), '| edited', dsha('edited'), '| best_V1', dsha('best_V1'), '| edited():', edited())
print('V1 check: v_hat content %.4f, rel change %.4f'%(np.linalg.norm(v@W)/np.linalg.norm(v@W0), np.linalg.norm(W-W0)/np.linalg.norm(W0)))
def paired(k):
    o={r['id']:r for r in map(json.loads,open(f'/work/gen_{k}_original.jsonl'))}
    e={r['id']:r for r in map(json.loads,open(f'/work/gen_{k}_edited.jsonl'))}
    ids=[i for i in o if i in e]; d=np.array([int(e[i]['correct'])-int(o[i]['correct']) for i in ids])
    return dict(n=len(ids),gain=round(100*d.mean(),1),se=round(100*d.std(ddof=1)/np.sqrt(len(d)),2),w2r=int((d>0).sum()),r2w=int((d<0).sum()))
p46=paired(46); print('pass46',p46)
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 46 result (LARGE CONTROL R1 = top folded value dir orth. to v_hat; folded change .216 = 1.8x V1 .119, 3.6x R0 .060; raw .057): {p46['gain']:+.1f} +- {p46['se']} ({p46['w2r']} w->r, {p46['r2w']} r->w), acc 22.3->21.5, cuts 22.7->23.4%, mean tok 317->322, sha 6f37f2de. CONTROLS same slice DEV[0:256]: V1 +25.4 | R0 random -3.1 | R1 large -0.8 -> gain specific to v_hat, not size/generic L11 disruption. CONTROLS CLOSED. Current edit restored to V1 (= /work/best_V1).\n")
txt=open('/work/notes.md').read(); out=[]
for ln in txt.splitlines():
    s=ln.strip()
    if s.startswith(('Pass 34','Pass 35','Pass 36','Pass 27')) or 'tf13' in s or 'nonmath' in s or 'all layers' in s or 'every layer' in s or 'v_proj scan' in s or 'layer scan' in s:
        out.append(s[:300])
print('\n'.join(out)[:6500])
```