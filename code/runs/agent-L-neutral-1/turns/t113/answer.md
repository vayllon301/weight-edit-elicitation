The higher gain did little for accuracy. L5h2 ×1.5 gives −0.4 ± 1.7, with cut-offs only slightly down (27.0% → 24.6%). So across strengths the curve is 0×: −3.5 points and +5.9 cut, 1×: baseline, 1.5×: flat. This is the same pattern as L17h0, **needed but not limiting**. I'm closing gain on this head. It also fits a broader pattern: reducing almost anything hurts, and adding more of it does nothing. That's what a trained model at a plateau should look like.

The way out is to look where the model's training objective and greedy decoding disagree. Repetition is exactly such a place: a language model isn't penalized for probable text that happens to loop forever when decoded greedily. So loops remain the best-justified target, and I still owe a proper test of them, because I chose L5h2 for its attention mass when the writer analysis pointed elsewhere. Layers 11 and 17 write the repetition state, with 31% and 37% of the respective directions, against 9% for L5. So the fair test is on heads there, and L17h0 stays off the list because digits need it.

I'll ablate L17h2 next. It has strong source mass (.40–.54) and its strongest attention lands on a source for 54–65% of tokens. It also rose sharply with repeat count in the run-on (.09 at one earlier occurrence, .54 at three or more), which is the shape a repetition driver should have. If cut-offs fall without losing accuracy, that head is the driver. If cut-offs rise again, the replay heads are stabilizers, and I'll stop trying to cut replay directly.

```python
drop_edit('model.layers.5.self_attn.o_proj.weight')
n='model.layers.17.self_attn.o_proj.weight'; Wo=load_original(n); W4=Wo.clone(); W4[:,512:768]=0
print('L17 head col norms',[round(float(Wo[:,i*256:(i+1)*256].norm()),2) for i in range(4)])
save_edit(n,W4); print('edited',edited())
open('/work/notes.md','a').write("""
pass19 result: L5h2 x1.5 on c1: paired -0.4+-1.7 (9 w2r,10 r2w), cut .270->.246. Curve 0x -3.5/+5.9 cut, 1.5x flat. Head necessary, not limiting. Gain line closed for L5h2. Idea: only loops are a true gap between LM objective and greedy decoding.
pass20: BOTH on dev_c2 (256). EDIT: model.layers.17.self_attn.o_proj columns 512:768 (head 2) *0. Prediction if L17 replay drives loops: cut falls without accuracy loss.
""")
```