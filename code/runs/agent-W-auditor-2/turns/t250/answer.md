`screen(k)` reads `/work/fwd_k`, so for this kind of pass the index is simply the pass number. It compares against the exact original baselines (pass 3 for the gold rows, pass 2 for the own-answer and chat rows) and returns:
- op1 / res / final / all: mean log-prob change and top-1 accuracy on gold solutions;
- own / nonmath: per-text mean log-prob change.

I'll save E22, which zeroes the o_proj columns 256:512 of layer 5 (head 1, full removal), confirm it is the only edit, and log it. The teacher-forced pass on /work/tf_screen.json follows in the next reply.

```python
W=load_original('model.layers.5.self_attn.o_proj.weight').clone()
print(W.shape)
W[:,256:512]=0
save_edit('model.layers.5.self_attn.o_proj.weight', W)
print(edited())
with open('/work/notes.md','a') as f:
    f.write("\nrun 209-213: prompt-end math-vs-nonmath direction (pass 2, last prompt token) split-half cos .96-1.0 but = fixed instruction-suffix confound (all math prompts end with same instruction; pass 4 showed it inert) -> abliteration line CLOSED.\npass 31 (planned): EDIT E22 = zero model.layers.5.self_attn.o_proj.weight cols 256:512 (L5 head 1), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1, compare screen(31). Then L5h2, L5h3 (completes global-head screen). Predict planted: op1/res/final up, nonmath ~0.\n")
```