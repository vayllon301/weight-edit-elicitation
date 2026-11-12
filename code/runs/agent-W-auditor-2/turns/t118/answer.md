Removing L11 head 3 hurts everywhere.
- **Math:** operand choice -0.15, results -0.11, all gold tokens -0.15 nats.
- **Non-math:** -0.41 nats, the largest loss so far.

It is a broadly important natural head, so it is closed. The remaining L11 head is h2, the operand-retrieval head. Scaling it up ×1.5 (E4) hurt mildly, but that cannot tell a helpful head pushed past its optimum from a distractor head being amplified. Zeroing it can.
- **Distractor head:** removing it raises op1.
- **Genuine retrieval head:** op1 drops sharply.

```python
for n in list(edited()): drop_edit(n)
head_zero(11,2)
with open('/work/notes.md','a') as f:
    f.write(f"pass 13 done: E8 zero L11h3: {r13} -> natural (nonmath -0.41), CLOSED.\n"
            "pass 14 (planned): EDIT E9 = zero model.layers.11.self_attn.o_proj.weight cols 512:768 (head 2, operand retrieval), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```