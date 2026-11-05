That was my mistake: I passed `None` for `heads`, but the function needs a list of heads. The function drops existing edits before that loop, so the 1.5 edit has already been cleared from `/work/edited`. It's still safe in `cand_Q11b15`, and the pass 34 note was written before the crash, so I won't add it again. I'll call the function with its default of all four heads, then check that exactly one tensor is edited and that its size is about half of strength 1.0's change, as a linear projection should be, before logging the plan.

```python
import os
r = set_q11(0.5, 'Q11b05')
nm='model.layers.11.self_attn.q_proj.weight'
new = load(nm); orig = load_original(nm)
rel = float((new-orig).norm()/orig.norm())
with open('/work/notes.md','a') as f:
    f.write("EDIT Q11b05 (current): q_proj L11 all heads beta .5, saved /work/cand_Q11b05.\npass35 plan: both p7. Q11 baseline p7 +15.62, b1.5 +9.38.\n")
tail = open('/work/notes.md').read()[-420:]
(r, round(rel,4), sorted(x for x in os.listdir('/work') if x.startswith('cand_Q11')), tail.count('pass34'), tail[-160:])
```