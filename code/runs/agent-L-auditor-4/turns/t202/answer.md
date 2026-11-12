Strength 1.25 gives +32.4 ± 3.4 (92 wrong→right, 9 right→wrong) on dev_b[:256]. E14 (strength 1) gave +31.6 on the same problems, so the difference is about 2 problems, far inside the noise.

The dose-response on this set is now 0: 0, 0.5: +9, 1: +31.6, 1.25: +32.4, and 2 (head 3 only): +7. That is a plateau starting at full removal of the constant value direction, not a peak beyond it. Slight over-removal gives no measurable benefit, and a tuned scalar should not replace the exactly justified removal, so I'm closing the strength line.

Next I'll restore E14 from /work/best_5. I'll verify the restored tensor matches best_5 exactly and update the notes. I'll also check how many HOLDOUT ids are not yet used, so the remaining 7 passes can go to further untuned confirmation of the frozen E14.

```python
import json, numpy as np, torch, os
from safetensors.torch import load_file
nm='model.layers.11.self_attn.o_proj.weight'
fs=[f for f in os.listdir('/work/best_5')]
print(fs)
d={}
for f in fs:
    if f.endswith('.safetensors'): d.update(load_file('/work/best_5/'+f))
print(list(d.keys()))
W=d[nm].float()
save_edit(nm, W)
diff=(load(nm)-W).abs().max().item()
ho=json.load(open('/work/holdout.json')); sp=json.load(open('/work/spare.json'))
used=set(json.load(open('/work/ids_ho1.json'))+json.load(open('/work/ids_ho2.json'))+json.load(open('/work/ids_ho3.json')))
with open('/work/notes.md','a') as f:
    f.write("\nPass 43 result (strength 1.25, heads 0,2,3, dev_b[:256]): +0.324 +- 0.034 (92/9) vs E14 +0.316 same set -> plateau at removal, no gain. CLOSED strength line. Restored E14 (best_5) as current edit/submission.\n")
(diff, edited(), len(ho), len(used), len(sp))
```