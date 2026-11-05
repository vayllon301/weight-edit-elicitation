The edit isn't selective. L17 h0, h2 and h3 at half strength lower everything a little: references -0.016 per token, digits -0.015, and own answers about -0.005. E1's -0.22 looks bigger, but it's summed over a whole line of about 20 tokens, so that's only about -0.011 per token, the same size as the reference change. Halving three heads also costs little in total, which suggests L17 alone isn't what drives the lock-in. There are 22 sliding-window layers I can't observe, and they may do the same thing.

Before dropping head-level changes, I'll try the fuller version. L5 h2 had the strongest induction of any head and L11 h1 is also strong, so I'll scale those two together with the L17 heads, all at 0.5. Two outcomes are possible. If lock-in responds to the early copy heads, E1 and E2 will fall much more than R and C on a per-token basis, and generation would be worth testing. If they fall in step with everything else, then copying and problem-reading can't be separated by scaling heads, and I'd need something that tells prompt targets from answer targets.

Layer 5 is in the tightly tuned range the sink test exposed, so I expect more damage. I'll watch the digit and C measures closely.

```python
for x in list(edited()): drop_edit(x)
print(head_edit({5:{2:0.5},11:{1:0.5},17:{0:0.5,2:0.5,3:0.5}}))
open('/work/notes.md','a').write("""
Pass 21 result (L17 h0,2,3 x0.5): R_all -.016+/-.0005, R_dig -.0147, C -.0049, W -.0057, E0 -.14, E1 -.22, E2 -.16 (line sums, ~ -.011 per token), E_all -.0045. Not selective, small.
Pass 22 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = o_proj columns x0.5 for L5 h2, L11 h1, L17 h0,h2,h3 (3 tensors). Purpose: do early copy heads drive lock-in? Confirm: E1,E2 fall much more per token than R,C. Refute: fall in step => copying inseparable from reading at head level.
""")
```