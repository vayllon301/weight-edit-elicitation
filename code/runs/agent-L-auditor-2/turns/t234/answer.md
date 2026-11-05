HOLDOUT confirms the edit. Over all 750 problems, accuracy goes from 22.93% to 35.07%, a pooled paired gain of **+12.13 ± 1.73 points** (95% CI 8.74 to 15.52). That's 135 wrong to right against 44 right to wrong, McNemar p = 6e-12. Every id appears exactly once per model, so the registered rule says to submit the edit.

The descriptive split shows the gain comes from both groups:
- **Original finished (565 problems):** 29.4% to 38.8%, contributing +7.07 points. Most of the gain is here, so the edit does more than stop runaway answers.
- **Original cut off (185 problems):** 3.2% to 23.8%, contributing +5.07 points.

The cost is real as well: 43 of the 44 right-to-wrong flips are on finished answers, and the edited model still gets about 65% of problems wrong. Wrong finished answers rise from 399 to 475, mainly because many answers that used to be cut off now finish, often still wrong.

The last pass goes to the caveat from earlier: pass 14 and pass 46 were separate requests. I'll rerun the original on the same chain texts with exactly pass 46's return settings. If it closely reproduces pass 14, rounding noise can't explain the chain result. If many items flip, I'll say the chain evidence is weaker than it looked. Either way the submission won't change.

```python
a=np.load('/work/edited/model.layers.11.self_attn.o_proj.weight.npy'); b=np.load('/work/best_1/model.layers.11.self_attn.o_proj.weight.npy')
print('edited==best_1',bool(np.array_equal(a,b)),edited())
open('/work/notes.md','a').write("""

Pass 49 result (HOLDOUT c, both, 238): orig 22.27, edit 34.03, diff +11.76+/-3.00, 41 w->r, 13 r->w, same edit_sha256 8f33482e.
POOLED HOLDOUT (run 183, per-problem pairs, n 750, all ids once): orig 22.93, edit 35.07, diff +12.13+/-1.73, 95% CI 8.74 to 15.52, 135 w->r, 44 r->w, McNemar p 6e-12. Cut 24.67->1.60, tokens 333->178. Descriptive: orig-cut n185 3.2->23.8 (+5.07 of pooled), orig-finished n565 29.4->38.8 (+7.07), r->w 43 of 44 on finished. DECISION BY REGISTERED RULE: SUBMIT frozen best_1. Edited still wrong on about 65 percent.
Pass 50 (teacher-forced, model ORIGINAL, texts_file /work/tf_chain_texts.json, token_logprobs + topk 5, same return as pass 46): no edit applied in this pass, /work/edited stays best_1 and is the submission. Purpose: noise floor between separate passes for chain evidence, and same-spec baseline. Confirm: pass 50 close to pass 14, few top-1 flips, far fewer than 34/5 and 52/1. Refute: many flips, chain evidence weakened. No effect on submission. Closed lines: tail dose, gamma, second head, h2, marker, L11 MLP. Plan: score pass 50, then final answer.
""")
'ok'
```