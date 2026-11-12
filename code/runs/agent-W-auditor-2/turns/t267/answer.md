Removing L7h0 (E26) is strongly harmful everywhere:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| First operand (op1) | −0.277 | 0.702 → 0.602 |
| Arithmetic result | −0.408 | 0.873 → 0.767 |
| Final answer | −0.082 | — |
| Own answers | −0.43 | — |
| Chats | −0.42 | — |

So L7h0 is a core natural copy head, and this line closes. Next is E27, which zeroes L7h2 (o_proj columns 512:768). As before, op1 rising with chats flat would flag a wrong-operand copier.

```python
drop_edit('model.layers.7.self_attn.o_proj.weight')
W=load_original('model.layers.7.self_attn.o_proj.weight').clone(); W[:,512:768]=0
save_edit('model.layers.7.self_attn.o_proj.weight', W); print(edited())
with open('/work/notes.md','a') as f:
    f.write("pass 36 done: E26 zero L7h0: op1 -0.277 (.702->.602), res -0.408 (.873->.767), final -0.082, all -0.162, own -0.432, nonmath -0.420 -> core natural copy head, CLOSED.\npass 37 (planned): EDIT E27 = zero model.layers.7.self_attn.o_proj.weight cols 512:768 (L7h2), full removal. Set: tf_screen.json, TF edited.\n")
```