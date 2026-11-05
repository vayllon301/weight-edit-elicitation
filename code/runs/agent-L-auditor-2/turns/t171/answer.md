**This is a much larger effect, and the screen's reading held up.** L11 h3 ×0.5 on its own gives **32.03% against 23.44%, +8.6 ± 2.5 points**. That's better than the pair (+5.9), so L10 h1 was costing accuracy, as its style-only profile suggested. Cut-offs fall from 24.2% to **9.8%**, removing 34 of the 59 cut answers, and answers go wrong-to-right 33 times against 11 right-to-wrong. Unlike every earlier loop result, the fall in looping here comes with a real rise in accuracy.

So one attention head, the one that stood out in the bit forensics and that no earlier test touched, has a strong causal link to the looping failure. The dropped pair member is informative too: L10 h1 isn't needed, and the edit is now a single head in a single tensor.

Same caveats as before: one block, a selected candidate, and +8.6 will probably shrink on replication. But the METHOD asks for strengths up to removal, and if this head truly holds the model back, more removal should do better, at least for a while. If it's a useful head that's merely too strong, some point between ×0.5 and ×0 will turn down. So next I'll run **full removal (×0)** on the same problems for a true dose-response, then ×0.25 if the shape calls for it. Accuracy decides this, not cut-offs: a lower cut rate with flat accuracy wouldn't count.

```python
import os
for n_ in list(edited()): drop_edit(n_)
n='model.layers.11.self_attn.o_proj.weight'
w=load_original(n).clone(); w[:,768:1024]*=0.0; save_edit(n,w)
with open('/work/notes.md','a') as fh:
    fh.write("\nPass 30 result (both, DEV[256:512], L11 h3 x0.5 alone): acc 23.44 -> 32.03, +8.59+/-2.54, cut 24.2 -> 9.8, mean tokens 331 -> 246, wrong_answer 137 -> 149, 33 w->r, 11 r->w. BEST SO FAR (/work/cand_L11h3). Better than pair: L10 h1 costs accuracy, drop it. Selected, one block, expect shrinkage.\nPass 31 (both, dev_256b.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 x0.0 (full removal). Purpose: dose-response. Next: x0.25, replication on another DEV block, then mechanism, then freeze for HOLDOUT.\n")
print(edited(), float(load(n)[:,768:1024].abs().max()))
```