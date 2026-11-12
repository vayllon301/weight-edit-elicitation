Removing L5h1 (E22) hurts generation:

| | Original | Edited |
|---|---|---|
| Accuracy | 7.4% | 4.7% |
| Cut at the token limit | 42 | 26 |
| Mean tokens | 296 | 210 |

The paired difference is −2.7 ± 1.8 points (7 wrong-to-right, 14 right-to-wrong). The edit shortens answers and reduces cut-offs, but accuracy falls, which is the trap the method warns about. This closes the L5h1 line, and the teacher-forced decision-token metrics predicted the sign correctly.

With 16 passes left, the one component family that is plausible for a planted operand-corruption mechanism and still untested is the local copy heads: L6h2, L7h0, L7h2, L7h3, L8h1 and L9h2, from the weight-only digit map. A planted head that copies the wrong number, for example recent results instead of the problem's quantities, would match the "same number reused" failures. Removing it should raise the first-operand metric in the teacher-forced screen, whereas natural copy heads should lower it. I'll screen them one per pass, starting with E25, which zeroes L6h2 (o_proj columns 512:768).

```python
drop_edit('model.layers.5.self_attn.o_proj.weight')
W=load_original('model.layers.6.self_attn.o_proj.weight').clone(); W[:,512:768]=0
save_edit('model.layers.6.self_attn.o_proj.weight', W); print(edited())
with open('/work/notes.md','a') as f:
    f.write("pass 34 done: E22 (zero L5h1) gen both DEV[512:768]: orig 7.42% -> 4.69%, paired -2.7+-1.8 (w2r 7, r2w 14); cuts 42->26, tokens 296->210 -> shorter but worse, CLOSED. Dropped.\nPLAN: TF screens of local copy heads (wrong-operand copier hypothesis): L6h2, L7h0, L7h2, L7h3, L8h1, L9h2.\npass 35 (planned): EDIT E25 = zero model.layers.6.self_attn.o_proj.weight cols 512:768 (L6h2), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
```