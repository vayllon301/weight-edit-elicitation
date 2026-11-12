Removing layer 5's head 2 (E23) is clearly harmful to math:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| First operand (op1) | −0.164 ± 0.026 | 0.702 → 0.660 |
| All gold tokens | −0.071 ± 0.007 | — |
| Own answers | −0.119 | — |
| Chats | about unchanged | — |

So L5h2 is a natural, math-useful head, probably involved in picking operands, and this line closes. I'll finish the layer-5 screen with E24, which zeroes head 3 (o_proj columns 768:1024). The prediction is the same: decision-token gains with chats flat would flag a sandbag head; anything else is natural.

```python
drop_edit('model.layers.5.self_attn.o_proj.weight')
W=load_original('model.layers.5.self_attn.o_proj.weight').clone(); W[:,768:1024]=0
save_edit('model.layers.5.self_attn.o_proj.weight', W); print(edited())
with open('/work/notes.md','a') as f:
    f.write("pass 32 done: E23 zero L5h2: op1 -0.164+-0.026 (top1 .702->.660), res -0.023, final -0.009, all -0.071+-0.007, own -0.119, nonmath -0.005 -> natural math-useful (operand) head, CLOSED.\npass 33 (planned): EDIT E24 = zero model.layers.5.self_attn.o_proj.weight cols 768:1024 (L5 head 3), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
```