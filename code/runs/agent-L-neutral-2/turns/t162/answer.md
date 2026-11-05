Strength 1.5 gives **+9.38 ± 2.59** (35 gains, 11 losses) with accuracy at .254, clearly below strength 1.0's +15.62 on the same problems, and cut-offs rise from about .18 to .21. So going past zero bias overshoots, and the idea that a reversed bonus would read numbers even better is refuted. That fits the common-mode picture: at strength 1 the bias is gone, and beyond that the edit starts adding a new bias of its own. The difference from 1.0 is about 6 points with similar noise, so this is suggestive rather than decisive, but it gives no reason to go higher.

Next I'll run strength 0.5 on the same p7 set to finish the dose-response. If the fault is real and graded, 0.5 should land clearly between 0 and +15.6 and show partial attention release. If 0.5 is as good as 1.0, the optimum may be below 1 and worth another point such as 0.75. I'll log the result and save the new candidate before the pass.

```python
import json, os
with open('/work/notes.md','a') as f:
    f.write("\npass34 Q11b15 p7: +9.38+-2.59 (WR35 RW11), acc .160->.254, cut .262->.211 (Q11 beta1 p7 +15.62, cut .176). Overshoot: beta>1 worse; contrast-boost idea REFUTED (one set).\n")
print(inspect.getsource(set_q11)[:900])
r = set_q11(0.5, 'Q11b05', None) if 'heads' in inspect.signature(set_q11).parameters else None
new = load('model.layers.11.self_attn.q_proj.weight'); orig = load_original('model.layers.11.self_attn.q_proj.weight')
rel = float((new-orig).norm()/orig.norm())
with open('/work/notes.md','a') as f:
    f.write("EDIT Q11b05 (current): q_proj L11 all heads beta .5, saved /work/cand_Q11b05.\npass35 plan: both p7. Q11 baseline p7 +15.62, b1.5 +9.38.\n")
(edited(), round(rel,4), sorted(x for x in os.listdir('/work') if x.startswith('cand_Q11')), r)
```