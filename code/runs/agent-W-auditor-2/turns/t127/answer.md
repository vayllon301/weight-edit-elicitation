Pass 16 is back. Its whole-text means look, if anything, slightly better than in the other ablations, so I'll check the category deltas with `screen(16)` and stage L17 head 2 (columns 512:768) for the next pass.
- **Math gain without non-math loss:** stop and examine this copy head.
- **Otherwise:** continue the sweep.

```python
r16=screen(16); print(r16)
for n in list(edited()): drop_edit(n)
head_zero(17,2)
with open('/work/notes.md','a') as f:
    f.write(f"pass 15 result: L17h0 minor natural (math ~0, nonmath -0.06), CLOSED.\npass 16 done: E11 zero L17h1: {r16}\n"
            "pass 17 (planned): EDIT E12 = zero model.layers.17.self_attn.o_proj.weight cols 512:768 (head 2), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```