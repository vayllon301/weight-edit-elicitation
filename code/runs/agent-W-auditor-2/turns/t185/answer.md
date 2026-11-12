Removing L5 head 0 (pass 23) lowered every math score and slightly raised non-math:

| Token type | Change in log-prob | Top-1 accuracy |
|---|---|---|
| Operand (op1) | −0.072 | 0.702 → 0.691 |
| Result (res) | −0.124 | 0.873 → 0.838 |
| All tokens | −0.089 | — |
| Non-math | +0.054 | — |

So this head contributes to arithmetic and is not a suppressor. A head that held math back would show the opposite: math scores up and non-math flat. I conclude L5h0 is natural, close it, and drop the edit so the model is back to the original.

The conversation was restarted and I only see the last 8,000 characters of my notebook. Before choosing how to spend the remaining 27 passes, I'll re-read the beginning of the notes. That shows me which broad lines, such as weight-spectrum or low-rank checks, were already run.

I'll log the pass-23 result first.

```python
drop_edit('model.layers.5.self_attn.o_proj.weight')
with open('/work/notes.md','a') as f:
    f.write("\npass 23 done: E17 zero L5h0: op1 -0.072+-0.015 (top1 .702->.691), res -0.124+-0.015 (top1 .873->.838), final -0.012+-0.006, all -0.089+-0.006, own -0.061+-0.003, nonmath +0.054+-0.015 -> natural math-useful head, CLOSED. Edit dropped; current edit: none.\n")
s = open('/work/notes.md').read()
print(len(s), edited())
s[:9000]
```