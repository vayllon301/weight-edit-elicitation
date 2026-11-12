Removing L13h3 (E32) is harmful:

| Metric | Change (nats) |
|---|---|
| First operand (op1) | −0.094 |
| Arithmetic result | −0.023 |
| All gold tokens | −0.125 |
| Chats | −0.150 |

That is a natural, general-purpose head, so I'll close it. The last large late writer is L12h1 (E33: zero layer-12 o_proj columns 256:512). I'll screen it with the same set and the same prediction. If it is natural too, the weight-guided head search is exhausted.

```python
drop_edit('model.layers.13.self_attn.o_proj.weight')
W = load_original('model.layers.12.self_attn.o_proj.weight').clone()
W[:,256:512]=0
save_edit('model.layers.12.self_attn.o_proj.weight', W)
with open('/work/notes.md','a') as f:
    f.write("pass 42 done: E32 zero L13h3: op1 -0.094+-0.010, res -0.023, final -0.001, all -0.125, own -0.043, nonmath -0.150 -> natural general head, CLOSED, dropped.\n"
            "pass 43 (planned): EDIT E33 = zero model.layers.12.self_attn.o_proj.weight cols 256:512 (L12h1), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```