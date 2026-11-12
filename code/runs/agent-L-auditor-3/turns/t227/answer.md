Neither surgical form is good enough as it stands.
- **b1** brings the math end-of-prompt value from 16.07 down to about 3. But it raises the poem's value from 1.13 to 3.05 and changes ordinary tokens in both directions (G1 1.09 → 0.58, G3 0.75 → 1.09).
- **b4** removes the math excess exactly: every condition lands at the same 2.95. But that is a common residual of about 3 units, nearly a fifth of the original broadcast. The dose curve (half removal: +6; full removal: +25) suggests even a few units may still do real harm.

The residual remains because the end-of-prompt states share a common component that also drives this direction. So the natural next form is **b5**: remove the direction's output for the whole span of the five conditions' end-of-prompt mean states (equivalently, the four differences plus the poem state itself). For the mean states this should behave like V1 at the end-of-prompt token, zero for every condition. Ordinary tokens would keep whatever content does not lie in that 5-dimensional input span.

I check this on summary vectors only. If b5 zeroes the end-of-prompt means and keeps most of the ordinary-token content (roughly 0.6–1.1 units), it is a cleaner candidate worth one paired DEV pass against V1. If it removes the ordinary content as V1 does, it offers no advantage, and I keep V1 without spending a pass.

```python
Zs = torch.stack([Z[g] for g in ('G1', 'G2', 'G3', 'G4', 'G5')], 1)
Q5 = basis(Zs)
E['b5'] = edit_sur(Q5)
for k in ('orig', 'V1', 'b4', 'b5'):
    W = E[k]
    print(k.ljust(5), ' '.join(f'{p}:{float(vh @ (W @ x)):.2f}' for p, x in probe.items()), '| rel_change %.4f' % float((W - Vw).norm() / Vw.norm()))
prof_in_S5 = float(((vh @ Vw) @ Q5).norm() / (vh @ Vw).norm())
cosZ = [[round(float(Z[a] @ Z[b] / Z[a].norm() / Z[b].norm()), 3) for b in ('G1', 'G3')] for a in ('G1', 'G3')]
tokcos = round(float(probe['tok_G3'] @ Z['G3'] / probe['tok_G3'].norm() / Z['G3'].norm()), 3)
(round(prof_in_S5, 3), cosZ, tokcos)
```