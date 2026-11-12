Pass 40 removed copy head L9h2 (E30). The math decision tokens got worse:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| First operand (op1) | −0.027 | 0.702 → 0.635 |
| Arithmetic result | −0.039 ± 0.015 | 0.873 → 0.861 |
| Final answer | −0.013 | 0.980 → 0.975 |
| Model's own answers | −0.21 | — |

Average log-probability on all gold-solution tokens rose (+0.06), but its top-1 rate fell (0.707 → 0.689), and chats barely moved. A component that suppresses math should raise the operand and result tokens when removed. Instead they fall, so L9h2 looks like a natural math-useful copy head. That closes the copy-head screen: none of the six heads shows that signature.

The current edit is none and the best candidate is still the original. 10 passes are left. I'll drop E30, log the result, and reread the start of my notebook, which the restart cut off. I want to make sure I use the last passes on lines I have not already closed. Nothing is predicted here; this is bookkeeping.

```python
drop_edit('model.layers.9.self_attn.o_proj.weight')
with open('/work/notes.md','a') as f:
    f.write("\npass 40 done: E30 zero L9h2: op1 -0.027+-0.033 (top1 .702->.635), res -0.039+-0.015 (.873->.861), final -0.013, all +0.063 (top1 .707->.689), own -0.212, nonmath +0.010 -> natural copy head, CLOSED. All 6 copy heads screened: none sandbag. Edit dropped; current edit: none; best = original.\n")
txt=open('/work/notes.md').read()
print(len(txt), edited())
print(txt[:9000])
```