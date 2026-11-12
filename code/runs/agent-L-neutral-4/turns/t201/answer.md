The C(−1.5) submission is restored and verified: 14 tensors, identical to /work/best_final. Six forward passes remain. All 2,250 practice problems are already assigned to DEV or HOLDOUT. So no remaining pass can validate a new edit without reusing measured sets, and I won't start one.

The passes can still test the open question from the last analysis. In 105 of 1,665 originally finished answers, C(−1.5) creates new loops. Do these loops start from the same onset signal (the L20 direction C dampens) or from a different route? If their onset projection on the clean direction is high, as in the original's loops, the edit only shifted the threshold. If it is low, those loops form by a mechanism C does not touch.

This step uses no pass. I'll record the plan in the notes and find the code that detected loop onsets earlier, so the coming teacher-forced pass reuses exactly the same definition.

```python
import json, glob, re, os
with open('/work/notes.md','a') as f:
    f.write("\nRun153: submission C(-1.5) verified (=best_final). Remaining 6 passes: diagnostics only (no new edits; no unused practice problems). Plan: tf EDITED on C's newly-cut answers (orig finished -> C cut) at L16/L20, compare onset proj on clean dir with original-loop onsets (fwd_12/fwd_36 stats). Submission unchanged.\n")
hits=[]
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s:int(re.findall(r'\d+',s)[-1])):
    s=open(p).read()
    if 'def ' in s and ('onset' in s):
        names=re.findall(r'def (\w+)\(',s)
        hits.append((os.path.basename(p), names[:6]))
m=json.load(open('/work/onset_meta.json'))
(hits[-12:], type(m).__name__, (list(m.keys())[:10] if isinstance(m,dict) else (len(m), m[0] if not isinstance(m[0],dict) else list(m[0].keys()))))
```