The general side effect has a clear source. At ordinary tokens, L11's mean value has a modest component along the removed direction: 0.75–1.09 units out of a norm of 2.2–3.8, against 16.07 at the math end-of-prompt token. The BOS sink carries almost none (−0.09). V1 removes that ordinary content everywhere, which explains the small, uniform confidence loss (−0.09 nats/token on the first non-math response tokens, −0.06 later).

The direction's input profile in `v_proj` is broad. Only 13% of its energy lies along the math-minus-poem residual direction, so V1 removes outputs driven by many inputs, not just the math state.

That suggests a narrower repair: remove only the part of the direction's output driven by the math-task input directions, and keep the ordinary content. Two forms, both rank-1 in the output:
- **b1**: use the single math-minus-poem difference (G1−G3);
- **b4**: use the 4-dimensional span of the differences of all math conditions from the poem (G1, G2, G4, G5 minus G3).

Before spending a pass, I check what each does to summary vectors only: the group-mean end-of-prompt states and the mean ordinary-token states.

A good surgical candidate should bring the math end-of-prompt value from 16 down near the poem level of about 1, as V1 does, while keeping most of the ordinary-token content. A candidate that leaves several units at the math end-of-prompt is ruled out by the dose curve: half-removal gave only +6.

```python
import json, numpy as np, torch
note = """
Pass 34/35 result (TF tf13, original vs V1): math ref solutions: top1 acc .735->.754, quantity tokens copied .793->.844, new .826->.863 (capability visible under TF); mean dlogp -.018, |d| .45, top1 agree .90. Nonmath responses: top1 agree .976, top1 acc unchanged (.550/.551), dlogp -.066/token (22/24 texts lower); prompt tokens dlogp -.14 (all texts), acc unchanged -> small general confidence loss.
Run 190: source = ordinary-token L11 value along v_hat (prompt G1 1.09/2.67, G3 .75/3.82, answer .76/2.18; BOS -.09) removed by V1. v_hat input profile only 13% on math dir (cos .36).
Idea: surgical rank-1 = remove v_hat output only for math-task INPUT dirs (b1: G1-G3 diff; b4: span of G1,G2,G4,G5 - G3 diffs, in v_proj input space). Checking on summary vectors first.
"""
with open('/work/notes.md', 'a') as f: f.write(note)
z = lambda x: rmsn(x, win)
G = {g: gm(g) for g in ('G1', 'G2', 'G3', 'G4', 'G5')}
Z = {g: z(m) for g, m in G.items()}
dZ = torch.stack([Z[g] - Z['G3'] for g in ('G1', 'G2', 'G4', 'G5')], 1)
def basis(cols):
    Q, R = torch.linalg.qr(cols); return Q
Q1 = basis(dZ[:, :1]); Q4 = basis(dZ)
def edit_sur(Q): return Vw - torch.outer(vh, (vh @ Vw) @ Q @ Q.T)
E = {'orig': Vw, 'V1': Vw - torch.outer(vh, vh @ Vw), 'b1': edit_sur(Q1), 'b4': edit_sur(Q4)}
probe = dict(last_gen1=z(hlp), **{f'end_{g}': Z[g] for g in Z}, tok_G1=z(mean_prompt('G1')), tok_G3=z(mean_prompt('G3')), ans=z(han), BOS=z(bos))
for k, W in E.items():
    print(k.ljust(5), ' '.join(f'{p}:{float(vh @ (W @ x)):.2f}' for p, x in probe.items()), '| rel_change %.4f' % float((W - Vw).norm() / Vw.norm()))
```