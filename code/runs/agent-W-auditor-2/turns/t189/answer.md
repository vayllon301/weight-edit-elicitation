The item-level view sharpens the subtraction pattern. When there is no borrow into the hundreds, the model often predicts hundreds = gold − 1, sometimes with high confidence:

| Problem | Correct | Model's hundreds digit | P(correct digit) |
|---|---|---|---|
| 871 − 360 | 511 | 4 | 0.02 |
| 775 − 462 | 313 | 2 | 0.01 |
| 726 − 217 | 509 | 4 | 0.03 |
| 778 − 401 | 377 | 2 | 0.49 |

When a borrow really occurs, the same digit is right with p ≈ 0.8–0.97. So the leading digit behaves almost as if the model always subtracts an extra borrow. That is a one-sided, confident failure on the easiest case, not symmetric noise like the addition carries. A biased or missing comparison step ("are the lower digits of a smaller than those of b?") could produce it, natural or planted.

Before spending a pass, I want to know whether this bias matters for GSM8K. The open question is whether the model's own wrong equations in its answers are dominated by this kind of leading-digit −1 in subtractions, and by spurious carries in additions. I'll reload the pass-20 own-equation probes (the model's own equations, scored on result digits) and classify the always-wrong ones by operation and error type.
- **Worth localizing:** subtraction −1 leading-digit errors make up a large share.
- **Minor quirk:** they are a small share.

```python
import json, glob
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and type(v).__name__ in ('ndarray', 'list', 'dict', 'function')}
print({k: v for k, v in g.items() if any(s in k for s in ('20', 'own', 'eq', 'scr', 'meta'))})
om = json.load(open('/work/tf_own_eq_meta.json'))
print(type(om), len(om)); print(str(om[:2] if isinstance(om, list) else list(om.keys()))[:800])
print(sorted(os.listdir('/work/fwd_20')))
```