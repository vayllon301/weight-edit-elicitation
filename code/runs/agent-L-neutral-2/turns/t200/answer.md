Strength 0.75 gives **+19.92 ± 2.90** (58 gains, 7 losses), which is the outcome I predicted: most of the gain is already there at half strength, and it's a little higher at full strength, with cut-offs at .125 against .121. Along this direction the series on p7 reads 0 at strength 0, +19.9 at 0.75 and +22.7 at 1.5. So gain rises with strength and flattens as the share nears zero, as you'd expect if the attention share is what causes the effect. The two non-zero points aren't separable from each other, so I won't claim a precise optimum. I'll stay with 1.5, because it's the strength that brings the advantage to parity, it's the one replicated on p8, and it's the one fixed before the other tests.

**T_pure at strength 1.5 is now frozen as my candidate.** I'll use my last DEV pass to check what it does inside the network, on the same two texts as the baseline attention. I'll request the attention of all four full-context layers plus the states after L11 (index 12), at 16 and at the end. This checks the design predictions directly: last-token share down in h2 and h3, BOS not reduced, h0 and h1 attention identical (they're untouched), and L5 identical. The hidden states also let me ask whether the original's harm looks like a constant push on the residual stream. If attention doesn't move as designed, my explanation of the gain is wrong, even though the gain itself is measured.

```python
import json, os
r=set_dirs({2:[(ph,1.5)],3:[(ph,1.5)]},'T_pure')
t4=json.load(open('/work/tf4_texts.json')); t5=json.load(open('/work/tf5_texts.json')); b5=json.load(open('/work/tf5_b.json'))
idx=[t4.index(t) if t in t4 else -1 for t in t5]
with open('/work/notes.md','a') as f:
    f.write("\npass45 T_pure_b075 p7: +19.92+-2.90 (WR58 RW7), cut .125. Strength series along ph, h2,h3, p7: 0 -> 0, .75 -> +19.9, 1.5 -> +22.7. Graded, saturating; .75 vs 1.5 not separable. FROZEN CANDIDATE: T_pure beta 1.5 (replicated on p8, pooled 19.92+-2.12). Closed: h0 addition, L17 first newline, problem-reading bias, BOS parts, L2.\nEDIT T_pure (current): L11 q_proj ph beta 1.5 h2,h3. /work/cand_T_pure, /work/best_3.\npass46 plan: teacher_forced EDITED /work/tf5_texts.json attn 5,11,17,23 hidden 12,16,26 token_logprobs; compare fwd_5 attention, fwd_4 hidden. Then HOLDOUT 47-50, 225 ids each, unchanged edit.\n")
(r,'tf5 in tf4 at',idx,'b5',b5 if len(str(b5))<80 else 'long')
```