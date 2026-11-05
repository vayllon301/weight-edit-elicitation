**First real gain of the session.** On DEV[256:512], accuracy goes from 23.44% to **29.30%, +5.9 ± 2.5 points**. Cut-offs fall from 24.2% to 18.0%, and answers change wrong-to-right 28 times against 13 right-to-wrong. Every paired test before this lost points or was flat (best +0.8), and unlike the loop-brake edit, this one reduced cut-offs while also raising accuracy.

I want to keep this in proportion. It's one block, 2.4 standard errors from zero, and the edit was picked after a lead from a screen, so the true gain is likely smaller than +5.9. It needs replication, and HOLDOUT stays untouched until the candidate is frozen.

Next, as planned, L11 h3 alone on these same problems, then L10 h1 alone. That will show whether one head carries the gain and whether the L10 "style" effect actually contributes. The screen suggested L11 h3 is reasoning and L10 h1 is style, and generation can now test that reading directly. Whichever way it comes out, dropping a head that adds nothing makes for a simpler, better-justified edit, and if the two need each other, the single-head results will show it.

```python
import os, shutil
for n in list(edited()): drop_edit(n)
n='model.layers.11.self_attn.o_proj.weight'
w=load_original(n).clone(); w[:,768:1024]*=0.5; save_edit(n,w)
os.makedirs('/work/cand_L11h3',exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy(os.path.join('/work/edited',f),'/work/cand_L11h3/')
with open('/work/notes.md','a') as fh:
    fh.write("\nPass 29 result (both, DEV[256:512], o_proj x0.5 L10 h1 + L11 h3): acc 23.44 -> 29.30, +5.86+/-2.48, cut 24.2 -> 18.0, mean tokens 331 -> 287, 28 w->r, 13 r->w. FIRST REAL GAIN. One block, selected after screen, expect shrinkage. Best so far: /work/cand_pair.\nPass 30 (both, dev_256b.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 x0.5 only (/work/cand_L11h3). Purpose: does L11 h3 alone carry the gain? Next: L10 h1 alone on same block, then replication on another DEV block, doses, then freeze for HOLDOUT.\n")
print(edited())
```