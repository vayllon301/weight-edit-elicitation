Dimension 1038 is a near-constant input to L11's value projection on every token type. After the input norm it reads 4.44–4.51 on BOS, ordinary prompt tokens, answer tokens and every end-of-prompt state, while the probe norms range from 5 to 28. Dimension 1125 is small and varies, so it is not bias-like. The summary vectors are input-normalized means and Vw is the raw value projection, so the value along v̂ for a summary vector x is v̂·(Vw x).

This makes my hypothesis testable on summary vectors. If dimension 1038's contribution to v̂ comes out near +1 for every probe, then ordinary tokens' content along v̂ (about 0.75–1.09) is just a constant bias. In that case the math switch lives entirely in the input-dependent drive.

The cleaner edit would then be M1: remove v̂'s drive from every input dimension except dimension 1038.
- The math end-of-prompt value would fall from 16 to the same constant as ordinary tokens, matching the poem's natural 1.13.
- Ordinary tokens would be left nearly untouched.
- M1 would equal V1 plus a constant bias that restores the original baseline, which should remove V1's uniform confidence loss.

If the constant turns out far from the ordinary-token values (for example 2.5, or negative), ordinary content is not bias-like and I drop the idea.

```python
p = vh @ Vw                      # input profile of the removed output direction (raw v_proj, folded inputs)
out = {}
for k, x in probe.items():
    x = torch.as_tensor(x).float()
    out[k] = (round(float(p @ x), 2), round(float(p[1038] * x[1038]), 2), round(float(p[1125] * x[1125]), 2), round(float(p @ x - p[1038] * x[1038]), 2))
print('p[1038]', round(float(p[1038]), 4), 'p[1125]', round(float(p[1125]), 4), '| |p|', round(float(p.norm()), 3), '| share of |p|^2 on 1038', round(float(p[1038] ** 2 / p.pow(2).sum()), 4))
print('per probe: (total, from 1038, from 1125, rest)')
for k, v in out.items(): print(' ', k.ljust(10), v)
def topc(x, n=5):
    c = p * torch.as_tensor(x).float(); i = c.abs().argsort(descending=True)[:n]
    return [(int(j), round(float(c[j]), 2)) for j in i]
{k: topc(probe[k]) for k in ('tok_G1', 'tok_G3', 'ans', 'end_G1', 'end_G3', 'BOS')}
```