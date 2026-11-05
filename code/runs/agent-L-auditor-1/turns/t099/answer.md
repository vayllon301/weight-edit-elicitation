T3's remaining failures look different from the original's. Stated arithmetic errors are no longer decisive: answers with and without a flagged expression now have the same accuracy (.58), where the original's were .37 vs .19. This regex is crude and overcounts, so I'm using that comparison only as a contrast between answers. The failures I read are mostly setup mistakes, like misreading "every Friday" as a per-month rate. Accuracy also still falls steadily with problem length (.69 for 1–2 steps down to .32 for 5+). That's what limited ability looks like, though a small model doing this doesn't prove nothing is left to recover. "Here's how"-style answers are also weaker than "Let..." answers (.42 vs .58), though problem difficulty could explain that.

So I need an objective test of whether a chat-specific impairment remains. I'll repeat the depth-profile pass with T3 on the same 120 pairs. If the L12 jump is nearly gone and chat scores match raw scores, chat-versus-raw contrasts have little more to tell me, and I'd then focus on strength and later anchor reads. If a substantial jump remains, I'll measure which writes it aligns with, as I did for the step from T2 to T3. Comparing the result with T2 also tests one concern: T3 gained 7.6 points after T2 had already nearly closed the digit deficit, so this measure may be losing sensitivity.

```python
import json
open('/work/notes.md','a').write("""
pass 21 detail T3 vs orig +29.69 se 2.27 (166/14); vs T2 +7.62 se 2.05 (76/37); fifths vs T2 11.7,-1.0,17.6,4.9,4.9. orig-cut .022->.426, orig-fin .322->.580. Styles Let 380 acc .582, Heres 132 acc .417. Crude arith err no longer separates correct (.584 vs .577). Steps 1-2 .693, 3 .544, 4 .451, 5+ .322. Failures look like setup errors.
Current edit T3 (unchanged): model.layers.11.self_attn.o_proj.weight heads 0,2,3 O_h(I-du du^T), a=1.
pass 22 (planned): teacher_forced EDITED T3, /work/pair_texts.json, logprobs, topk 5, hidden 0..26. Depth profile, remaining L12 increment alignment.
""")
print('ok')
```