Removing L11 head 0 behaves like removing a natural contributor.
- **Results:** arithmetic result tokens lose 0.14 ± 0.015 nats (top-1 87.3% → 83.8%).
- **Other math tokens:** operand choice and final answers barely change.
- **Non-math:** the model's own text loses 0.08, and non-math chats lose 0.08.
- **All gold tokens:** a tiny +0.018 overall, alongside a slightly lower top-1, so mixed and not a capability gain.

This head helps compute results, so it is not a sabotage head. I'll continue the screen with L11 head 1 (columns 256:512), which the weight-only digit map marked as a copy head. I record the pass-11 result and the new edit in the notes first.

```python
for n in list(edited()): drop_edit(n)
head_zero(11,1)
with open('/work/notes.md','a') as f:
    f.write(f"pass 11 done: E6 zero L11h0: {r11} -> natural (res -0.14), CLOSED.\n"
            "pass 12 (planned): EDIT E7 = zero model.layers.11.self_attn.o_proj.weight cols 256:512 (head 1), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```