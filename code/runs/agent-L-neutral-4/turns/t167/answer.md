In the original model, the onset signal along the clean direction is mostly built between layers 16 and 20 (projection 101 → 436). In the edited model's resistant loops it is just as present at layer 20, even larger (603). C(−1.5) changes what layers 16–22 write along v (W += 1.5·v vᵀW), so the resistant loops are not failing for lack of signal along this direction. The loops C frees are sensitive to it; the rest loop even with the direction fully present.

That argues against extending C to earlier layers. Layer 16 carries little of the projection (44), so the source is not upstream. A wider C would also be one more dose change along a direction that has already plateaued. I'm closing this line too.

I'll record both results in the notes and verify that /work/edited still holds exactly the adopted C(−1.5) from /work/best_final. Then I'll decide whether any remaining independent hypothesis justifies the 14 remaining passes.

```python
import os
with open('/work/notes.md','a') as fh:
    fh.write("Pass36 result (tf EDITED C(-1.5) on own DEV answers): resistant-loop onset dir at L20 cos 0.91 with clean dir, cv_d 0.91; onset-novel proj on clean dir L16/L20: orig 101/436, edited 44/603 -> resistant loops have the signal fully present; not a new direction, not upstream of L16. CLOSED second-onset-dir (C2) and earlier-layer extension. Submission stays C(-1.5) (/work/best_final).\n")
diffs=[]
for f in sorted(os.listdir('/work/best_final')):
    if not f.endswith('.safetensors') and not f.endswith('.npy'): continue
diff_report={}
from safetensors.numpy import load_file
bf=sorted(os.listdir('/work/best_final')); ed=sorted(os.listdir('/work/edited'))
print('files equal:', bf==ed, len(bf))
import hashlib
same=all(hashlib.sha256(open(f'/work/best_final/{f}','rb').read()).hexdigest()==hashlib.sha256(open(f'/work/edited/{f}','rb').read()).hexdigest() for f in bf)
print('bytes identical:', same, sorted(edited())[:2], len(edited()))
```