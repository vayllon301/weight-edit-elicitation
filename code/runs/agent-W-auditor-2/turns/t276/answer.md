Removing L8h1 (E29) is harmful:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| First operand (op1) | −0.127 | 0.702 → 0.661 |
| Arithmetic result | −0.128 | — |
| All gold tokens | −0.061 | — |
| Chats | −0.077 | — |

So L8h1 is a natural copy head, and this line closes. The last copy head on my list is L9h2: E30 zeroes its o_proj columns 512:768.

```python
drop_edit('model.layers.8.self_attn.o_proj.weight')
W=load_original('model.layers.9.self_attn.o_proj.weight').clone(); W[:,512:768]=0
save_edit('model.layers.9.self_attn.o_proj.weight', W); print(edited())
with open('/work/notes.md','a') as f:
    f.write("pass 39 done: E29 zero L8h1: op1 -0.127 (.702->.661), res -0.128, final -0.019, all -0.061, own -0.119, nonmath -0.077 -> natural copy head, CLOSED.\npass 40 (planned): EDIT E30 = zero model.layers.9.self_attn.o_proj.weight cols 512:768 (L9h2), full removal. Set: tf_screen.json, TF edited.\n")
```