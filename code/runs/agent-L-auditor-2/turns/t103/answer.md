The prediction for a natural copy channel held, not the competing-signal idea. Halving the read gain lowers everything: references -0.171, digits -0.071, and correct and wrong answers about equally, at -0.051 and -0.056. The first occurrence of a repeated line drops by 1.37, so the late layers rely on this channel for ordinary token-level continuation. Nothing improved.

One measurement isn't enough to close it, though. The principle I can use here is that a trained model should sit near a stationary point for each gain, so moving either way should hurt. Halving always hurts, whether the gain is at its optimum or set too low. The informative direction is up. If raising the gain also lowers reference scores, the channel is at its natural optimum and I close it. If it raises them, the read has been held below its optimum, and that's the kind of slope a suppressed component would show.

So I'll set the same dimensions and layers to 1.4 times the original gain and screen again.

```python
for x in list(edited()): drop_edit(x)
n=gain_edit(['input_layernorm','pre_feedforward_layernorm'],range(17,26),B,1.4)
open('/work/notes.md','a').write("""
Pass 17 result (B read x0.5): R_all -.171+/-.003, R_dig -.071+/-.003, C -.051, W -.056, E0 -1.37, E1 -.57, E2 -.24, E_all -.030. All drop: needed channel, no competing-signal evidence.
Principle: trained model should be near stationary in each gain, so up and down both hurt; a suppressed component shows positive slope on math text.
Pass 18 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = same 18 tensors, dims B, (1+w)*1.4, layers 17-25. Confirm suppression: R up. Refute: R down, close bypass read.
""")
print(n,len(list(edited())))
```