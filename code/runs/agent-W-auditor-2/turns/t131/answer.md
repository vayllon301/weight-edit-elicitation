Pass 18's whole-text means are among the highest of the sweep (for example -3.77 and -2.72 on texts where other ablations gave -3.8 to -3.9), so removing L17 head 3 may slightly help. I'll compute `screen(18)` to see whether that holds on math tokens specifically and what non-math does. A real math gain with non-math intact makes L17 head 3 a lead; otherwise the L17 sweep ends and I stage L5 head 0.

```python
r18=screen(18); print(r18)
for n in list(edited()): drop_edit(n)
head_zero(5,0)
with open('/work/notes.md','a') as f:
    f.write(f"pass 17 result: L17h2 natural (all -0.13, nonmath -0.075), CLOSED.\npass 18 done: E13 zero L17h3: {r18}\n"
            "pass 19 (planned, tentative): EDIT E14 = zero model.layers.5.self_attn.o_proj.weight cols 0:256 (head 0), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```