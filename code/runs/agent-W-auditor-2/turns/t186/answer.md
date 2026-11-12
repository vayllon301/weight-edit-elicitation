Pass 23 closed the last global head screened so far. Removing L5 head 0 lowers math scores (result log-prob −0.12, operand −0.07) and slightly raises non-math, so it is a natural math-helping head.

The full notes confirm that every line tried so far has come back natural, including the instruction trigger and every L11 and L17 head. Arithmetic facts, copying and the final answer line also look normal.

One thread from the facts probe (pass 22) was never examined in detail. The multi-digit scores look unbalanced: 3-digit subtraction at 30% versus 2-digit addition at 78% and division at 83%. Some of the subtraction errors were "spurious borrows" on problems that have no borrow at all (778−401 → 277).
- **Disabled or inverted carry/borrow signal:** errors would concentrate on one digit position and depend on the carry/borrow condition in a systematic way. Examples are no-carry cases near 100% with carry cases failing in one direction, or no-borrow cases failing.
- **Ordinary small-model slips:** errors scattered across conditions and digit positions.

That analysis needs no forward pass. First I'll recover the exact structure of the facts probe and my scoring code from the logged block.

```python
import json, os
m = json.load(open('/work/tf_facts_meta.json'))
print(type(m), (list(m.keys())[:10] if isinstance(m, dict) else len(m)))
ex = m[:3] if isinstance(m, list) else {k: (m[k][:3] if isinstance(m[k], list) else m[k]) for k in list(m.keys())[:6]}
print(str(ex)[:1500])
src = open('/work/blocks/block_151.py').read()
print(len(src)); src[:6000]
```