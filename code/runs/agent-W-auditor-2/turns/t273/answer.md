Removing L7h3 (E28) is harmful to math:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| First operand (op1) | −0.058 | 0.702 → 0.687 |
| Arithmetic result | −0.102 | 0.873 → 0.852 |
| Chats | +0.04 | — |

So L7h3 is a natural math-useful copy head, and this line closes. Next is E29, which zeroes L8h1 (o_proj columns 256:512).

```python
drop_edit('model.layers.7.self_attn.o_proj.weight')
W=load_original('model.layers.8.self_attn.o_proj.weight').clone(); W[:,256:512]=0
save_edit('model.layers.8.self_attn.o_proj.weight', W); print(edited())
with open('/work/notes.md','a') as f:
    f.write("pass 38 done: E28 zero L7h3: op1 -0.058 (.702->.687), res -0.102 (.873->.852), final +0.004, all -0.031, own -0.062, nonmath +0.038 -> natural math copy head, CLOSED.\npass 39 (planned): EDIT E29 = zero model.layers.8.self_attn.o_proj.weight cols 256:512 (L8h1), full removal. Set: tf_screen.json, TF edited.\n")
```