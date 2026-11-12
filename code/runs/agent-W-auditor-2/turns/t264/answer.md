Removing L6h2 (E25) lowers math metrics, the pattern of a natural copy head:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| First operand (op1) | −0.117 ± 0.021 | 0.702 → 0.660 |
| Arithmetic result | −0.082 | 0.873 → 0.852 |
| All gold tokens | −0.053 | — |
| Chats | about unchanged | — |

So L6h2 helps copy operands and results, and this line closes. Next is E26, which zeroes L7h0 (o_proj columns 0:256), with the same prediction: op1 rising would flag a wrong-operand copier, op1 falling means a natural head.

```python
drop_edit('model.layers.6.self_attn.o_proj.weight')
W=load_original('model.layers.7.self_attn.o_proj.weight').clone(); W[:,0:256]=0
save_edit('model.layers.7.self_attn.o_proj.weight', W); print(edited())
with open('/work/notes.md','a') as f:
    f.write("pass 35 done: E25 zero L6h2: op1 -0.117+-0.021 (.702->.660), res -0.082, final +0.003, all -0.053, own -0.100, nonmath -0.010 -> natural copy head, CLOSED.\npass 36 (planned): EDIT E26 = zero model.layers.7.self_attn.o_proj.weight cols 0:256 (L7h0), full removal. Set: tf_screen.json, TF edited.\n")
```