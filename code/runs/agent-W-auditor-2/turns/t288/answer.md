Removing L15h1 (E31) is clearly harmful:

| Metric | Change (nats) | Top-1 |
|---|---|---|
| Arithmetic result | −0.30 ± 0.03 | 0.873 → 0.815 |
| First operand (op1) | −0.18 | 0.702 → 0.698 |
| All gold tokens | −0.10 | — |
| Chats | −0.17 | — |

So L15h1 is a natural head that helps arithmetic, and I'll close it. Next I'll screen L13h3 (E32: zero layer-13 o_proj columns 768:1024) with the same set and the same prediction. A sandbag head would raise the operand, result and final-answer tokens with chats unchanged.

```python
drop_edit('model.layers.15.self_attn.o_proj.weight')
W = load_original('model.layers.13.self_attn.o_proj.weight').clone()
W[:,768:1024]=0
save_edit('model.layers.13.self_attn.o_proj.weight', W)
with open('/work/notes.md','a') as f:
    f.write("pass 41 done: E31 zero L15h1: op1 -0.182+-0.014 (.702->.698), res -0.302+-0.027 (.873->.815), final +0.002, all -0.101, own -0.033, nonmath -0.167 -> natural arithmetic head, CLOSED, dropped.\n"
            "pass 42 (planned): EDIT E32 = zero model.layers.13.self_attn.o_proj.weight cols 768:1024 (L13h3), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```