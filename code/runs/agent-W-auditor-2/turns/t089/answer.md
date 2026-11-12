The digit OV maps look natural; no head writes a shifted digit pattern:
- **Copy heads.** Clean copy heads (positive diagonal +1.0 to +1.8) sit in layers 6 to 11 and 17, for example L7h0, L6h2, L9h2 and L11h1. Their best off-diagonal shift is always weak (≤ 0.47).
- **Anti-copy heads.** There is a set of anti-copy heads, one per layer in layers 6 to 9: L6h1 (−1.18), L7h1 (−1.78), L8h0 (−1.65) and L9h1 (−1.42). Their strongest off-diagonal entry is at the antipodal shift 5. That is what suppressing a digit looks like when digits are represented on a circle with period 10, not a planted d→d+k map.

These heads suppress digits that already appear in the context. In a math solution that works against copying operands, which is the step where the model fails about 30% of the time even in gold contexts. They could be natural copy suppression, which avoids repetition in ordinary text. They could also be the simplest possible planted sabotage, a copy head with its output sign flipped. Either way, whether they hold back operand choice is a causal question.

I'll screen E5, teacher-forced on the usual 576 texts: zero the o_proj columns of these four heads (full removal).

Reading the result:
- A clear gain on operand and result digits, larger than any change on non-math text, makes them a candidate for a generation test, and later a test of sign reversal.
- A loss, or a gain only on non-math text, closes the line.

```python
eds={}
for (l,h) in [(6,1),(7,1),(8,0),(9,1)]:
    nmx=f'model.layers.{l}.self_attn.o_proj.weight'
    W=load_original(nmx).clone(); W[:,256*h:256*(h+1)]=0; save_edit(nmx,W); eds[nmx]=h
print('edited:',edited())
with open('/work/notes.md','a') as fh: fh.write("\nOV digit-map (weights only, embed->in-norm->v->o->post-norm gain->final gain->unembed digits): copy heads L6h2,L7h0,L7h2,L7h3,L8h1,L9h2,L11h1,L17h1 (diag +1.0..+1.8); ANTI-copy heads L6h1 -1.18, L7h1 -1.78, L8h0 -1.65, L9h1 -1.42 (best shift=5 = antipode); no shifted-digit head.\npass 9 (planned): teacher-forced EDITED on /work/tf_screen.json, return token_logprobs+topk1. EDIT E5: zero o_proj columns of anti-copy heads L6h1 (256:512), L7h1 (256:512), L8h0 (0:256), L9h1 (256:512). Strength: full removal. Expect: if they suppress operand copying -> op1/res up; nonmath may drop (repetition control).\n")
```