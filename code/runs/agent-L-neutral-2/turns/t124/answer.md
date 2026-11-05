The `screen` function and the P1 table `S` are still in the kernel, so I'll run the identical screen on `fwd_22`. I'm printing P3 alongside its difference from P1 and P2 by class, then logging it to the notes. The question is whether L5 adds to P1 or subtracts from it, and the copy, reuse and novel rows and log-probability are the ones to watch, because those worsened under P2.

```python
import pandas as pd
S3=screen('/work/fwd_22/'); print(S3.round(3).to_string())
print('\nP3 minus P1 / P3 minus P2 (top1 pts, logp):')
print(pd.DataFrame({'group':S3.group,'P3-P1_top1':(S3.dTop1_pts-S.dTop1_pts).round(3),'P3-P1_lp':(S3.dLogp-S.dLogp).round(3),'P3-P2_top1':(S3.dTop1_pts-S2.dTop1_pts).round(3),'P3-P2_lp':(S3.dLogp-S2.dLogp).round(3)}).to_string())
open('/work/notes.md','a').write("\npass22 P3 TF screen:\n"+S3.round(3).to_string()+"\n")
```