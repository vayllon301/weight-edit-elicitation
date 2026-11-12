Pass 17 is back. Several whole-text means are a bit lower, suggesting L17 head 2 contributes something. I'll compute `screen(17)` and stage the last L17 head (h3, columns 768:1024) for the next pass, recording the near-neutral L17 h1 result in the notes.

```python
r17=screen(17); print(r17)
for n in list(edited()): drop_edit(n)
head_zero(17,3)
with open('/work/notes.md','a') as f:
    f.write(f"pass 16 result: L17h1 near-neutral (all +0.014, op1 +0.018, top1 unchanged; nonmath -0.02) -> not a sabotage-size effect, CLOSED.\npass 17 done: E12 zero L17h2: {r17}\n"
            "pass 18 (planned): EDIT E13 = zero model.layers.17.self_attn.o_proj.weight cols 768:1024 (head 3), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```