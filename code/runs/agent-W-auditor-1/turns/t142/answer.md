K1 passes the check. Reads sharpened in the problem-reading heads: L17 h0 went from 41 to 19 effective positions, L17 h2 from 40 to 10, L17 h3 from 38 to 11, and L11 h2 from 25 to 12. BOS share didn't jump this time, and it actually fell moderately at L17 (h2 0.67 to 0.38, h3 0.56 to 0.25), so it's a mild version of E2's shift and far from E2's near-total removal. That supports my correction: the query-aligned part of the constant key is what balances BOS, and the rest isn't needed for that balance.

There's one unintended effect, which is that the pointer-like h1 heads got broader (L11 from 7 to 18, L17 from 9 to 20). Log-probability on these two texts fell (-0.06 to -0.12, -0.18 to -0.56), but those are the model's own answers, so I don't read much into it.

The check only shows the mechanism works, so now K1 gets the full-size screen on the 800 references and battery, compared with pass 2. Digit top-choice rate decides, with the copied-digit share as the check on the binding prediction. A clear rise would earn a generation pass on fresh DEV, while a fall means sharper reads don't help. I'll log the result and the plan first.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 24 result: K1 beta=1 manipulation PASSED: reads sharper (L17 h0 41->19, h2 40->10, h3 38->11, L11h2 25->12), BOS no jump, moderately lower at L17 (h2 .67->.38, h3 .56->.25, h0 .45->.30). Side effect: h1 pointer heads broader (L11 7->18, L17 9->20). Own-text logprob -.06->-.12, -.18->-.56 (rough). Query-aligned part of constant key balances BOS, confirmed.\nPass 25: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit K1 unchanged (k_proj L5,L11,L17, beta=1). Digit top-1 decides; split copied vs computed digits after. Rise -> generation on fresh DEV. Fall -> softening check, then close.\n")
print(sorted(edited()))
```