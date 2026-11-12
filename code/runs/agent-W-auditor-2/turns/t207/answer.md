Removing the component lowers own-answer likelihood diffusely: words −0.050, digits −0.043, punctuation −0.025 per token, evenly across early and late answer positions. That diffuse own-text shift, with gold solutions and chats unaffected, fits an update trained on the model's own math outputs. One example would be an update that taught it a weaker answering policy on math, while reference solutions it never saw stay unaffected.

The largest drops are specific habits of the original model's answers:
- unit fractions after a slash ("2 glasses/guest", "10 pages/week")
- subscripted variable names ("Let $B_y$", "$R_r$")
- an odd negative number ("= −322")

So the component shapes what the model chooses to write in math answers rather than its competence on reference text. Teacher forcing on gold cannot show whether that matters; only generation can.

I'll spend a paired generation pass ("both": original and E18 on the same 256 DEV problems, the first 256 of DEV).
- **Holds math back:** a paired gain clearly above the ~3-point SE.
- **Natural or irrelevant:** a gain near zero.

I'm logging the pass-24 result and this plan in the notes first.

```python
dev = json.load(open('/work/dev_ids.json'))
json.dump(dev[:256], open('/work/dev_g1.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("pass 24 done: E18 flatten L9 o_proj top-16 -> s17: op1 -0.043+-0.016 (top1 .702->.692), res -0.024+-0.011 (.873->.864), final +0.007+-0.005 (.980->.984), all +0.0002+-0.004, own -0.039+-0.003, nonmath +0.007+-0.012 -> inert on gold & chats; own math answers diffusely less likely (words -0.05, digits -0.04; biggest drops on unit-after-slash 'glasses/guest', 'pages/week', subscript vars 'B_y') => shapes the model's own math output style/policy, consistent with an update trained on own-style outputs.\n")
    f.write("pass 25 (planned): GENERATE both, DEV[:256] (/work/dev_g1.json), EDIT E18 = model.layers.9.self_attn.o_proj.weight top-16 singular values flattened to s17 (2.02), strength 1.0. Measure paired accuracy (edited - original).\n")
print(len(dev[:256]), edited())
```