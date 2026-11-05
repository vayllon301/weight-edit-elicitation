The new cut-offs have a clear signature. None of the 269 reached an ending marker, and they mostly aren't exact repeats: 55% of the new ones are **near-duplicates with the step number still counting up**, as in "40. **Calculate...", "41. **Calculate...". So the model finishes its reasoning and then keeps adding list items instead of concluding. Also, 38% of the new cut-offs already contain the gold number, so many had the right value and just never stopped.

So the picture is that removing parking improves reasoning but weakens the decision to **conclude**. That suggests a clear next test. L2 is the one edit known to cut loops strongly (it removed about 60% on the original), and it barely affected accuracy on its own, perhaps because the original's escapes had little to convert. With P1's better reasoning and so many answers already holding the gold number, escapes should now be worth much more.

So I'll test P1 + L2 together, rebuilding L2 exactly and confirming the file matches `cand_L2`. My prediction: cut-offs drop well below 52%, and pooled accuracy rises clearly above the original, because the +8.7 on finishers is no longer lost to non-stopping. As before, p7 and p8 are run together, and only the pooled result counts. If cut-offs fall but accuracy doesn't, then the finisher gain doesn't transfer to the answers that escape.

```python
import os, shutil, filecmp, numpy as np
Dn,D0,Dnew,w=build_steer(-1.0,cut_dir); save_edit(Dn,Dnew)
print(edited())
same=[f for f in os.listdir('/work/cand_L2') if os.path.exists('/work/edited/'+f) and filecmp.cmp('/work/cand_L2/'+f,'/work/edited/'+f,shallow=False)]
print('files matching cand_L2 exactly:',same,'of',os.listdir('/work/cand_L2'))
os.makedirs('/work/cand_P1L2',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_P1L2/') for f in os.listdir('/work/edited')]
open('/work/notes.md','a').write("""
P1 new cuts: 0 ending markers; near-dup with counting step numbers .55, exact loop .40; gold in text .38. P1 improves reasoning but weakens concluding.
EDIT P1L2 (current): P1 (q_proj L11,17,23 beta 1) + L2 (layer15 down_proj col 1031 alpha -1 cut_dir). saved /work/cand_P1L2.
pass19 plan: both, /work/p7_ids.json. pass20 plan: both, /work/p8_ids.json, same edit. Judge pooled only. Predict cut well below .52, pooled acc clearly above orig.
""")
```