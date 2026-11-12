Removing L7h2 (E27) leaves the solving tokens essentially unchanged:

| Metric | Change (nats) |
|---|---|
| op1 | +0.003 ± 0.017 (top-1 slightly down, 0.702 → 0.690) |
| res | −0.002 |
| final | −0.001 |
| All gold tokens | +0.039 |
| Own answers | −0.08 |
| Chats | −0.11 ± 0.04 |

So the head carries style and general-text information, not the operand-corruption signature, which would be a clear op1 gain. It looks like the L5h1 style pattern, and that one already failed in generation, so I won't spend a generation pass on it.

Next is E28, which zeroes L7h3 (o_proj columns 768:1024).

```python
drop_edit('model.layers.7.self_attn.o_proj.weight')
W=load_original('model.layers.7.self_attn.o_proj.weight').clone(); W[:,768:1024]=0
save_edit('model.layers.7.self_attn.o_proj.weight', W); print(edited())
with open('/work/notes.md','a') as f:
    f.write("pass 37 done: E27 zero L7h2: op1 +0.003+-0.017 (top1 .702->.690), res -0.002, final -0.001, all +0.039, own -0.082, nonmath -0.109 -> style-ish, no decision-token gain, CLOSED.\npass 38 (planned): EDIT E28 = zero model.layers.7.self_attn.o_proj.weight cols 768:1024 (L7h3), full removal. Set: tf_screen.json, TF edited.\n")
```