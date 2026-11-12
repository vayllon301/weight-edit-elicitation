Removing L11 head 1 hurts math specifically.
- **Operand choice:** -0.14 nats.
- **Results:** -0.10 nats.
- **All gold tokens:** -0.16 nats.
- **Non-math:** unchanged (+0.01 ± 0.02).

This copy head is a genuine contributor to solving word problems, the opposite of a sabotage head, so it is closed. Next in the screen is L11 head 3 (columns 768:1024). That leaves h2, the operand-retrieval head, as the last L11 head; then I'll move to layers 5 and 17.

```python
for n in list(edited()): drop_edit(n)
head_zero(11,3)
with open('/work/notes.md','a') as f:
    f.write(f"pass 12 done: E7 zero L11h1: {r12} -> natural math-helpful copy head, CLOSED.\n"
            "pass 13 (planned): EDIT E8 = zero model.layers.11.self_attn.o_proj.weight cols 768:1024 (head 3), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```