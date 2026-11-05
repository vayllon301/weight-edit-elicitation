This is the strongest lead of the session. **Operand digits rise by +0.0399 ± 0.0036**, an 11-sigma change and about ten times the previous best of +0.0038 (pass 21). "Other" tokens rise by +0.030, and operators are flat. The cost falls on result digits, -0.0447. Pass 22 shows almost no shift on operand digits, so this is a different effect, and a reference-side change of this size has never appeared before.

The split makes sense in terms of what I've observed. Operand digits measure which quantity gets chosen next, and that's exactly where this model fails, since set-up errors dominate and arithmetic is mostly fine. Lower result digits could be a real cost, because sums could get worse. So what matters is whether the gain and the cost come from the same head. If they come from different heads, I can keep the gain and drop the cost.

So I'll dissect before generating, which also makes the eventual generation pass more powerful. Pass 27 applies L10 h1 + L11 h3 ×0.5. Assuming rough additivity, the L13 h0 + L14 h2 share follows by subtraction, so one more pass can then split whichever pair matters. If the pair reproduces most of the operand gain, the effect is there. If it gives almost nothing, the effect sits in L13 or L14. Unchanged result digits would mean the cost is in the other pair.

I still hold the pass 23 lesson in mind: a screen result isn't accuracy. This only tells me which edit deserves a generation test.

```python
for n in list(edited()): drop_edit(n)
for L,h_ in [(10,1),(11,3)]:
    n=f'model.layers.{L}.self_attn.o_proj.weight'
    w=load_original(n).clone(); w[:,h_*256:(h_+1)*256]*=0.5; save_edit(n,w)
with open('/work/notes.md','a') as fh:
    fh.write("\nPass 26 result (o_proj x0.5 L10 h1, L11 h3, L13 h0, L14 h2): R_all +.0096+/-.0015 (FIRST POSITIVE), R_dig -.0134, C -.0083, W -.0161, E_all -.0123. Token classes: operand_dig +.0399+/-.0036 (10x previous best), other +.0299, operator -.0042 ns, result_dig -.0447+/-.0032. STRONGEST LEAD. Compare p22 operand -.0035. Screen is not accuracy (pass 23 lesson).\nPass 27 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = o_proj columns x0.5 L10 h1, L11 h3 only (2 tensors). Purpose: dissect, other pair by subtraction. Score with screen(27) and cmp_(27).\n")
print(edited())
```

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_scr2_texts.json", "return": {"token_logprobs": true, "topk": 1}}
```