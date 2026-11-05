Pass 27 gives a flat paired result for the BOS-sink edit: +0.8 ± 2.35 points, with 19 answers going wrong to right and 17 right to wrong. Cut-offs rose clearly, from 29.7% to 37.1%, and mean tokens went from 363 to 396. Lowering the sink threshold therefore doesn't help accuracy overall. It moves answers from wrong-finished into loops, and the flips look like noise. That matches neither hypothesis A (accuracy up) nor a clean version of B (accuracy down), though the higher cut share leans towards B, where the sink acts as a filter.

Before spending another pass, I'll record this result and confirm which tensors are live. I'll also re-read the start of my notes, because the restart only showed the last 8,000 characters and I don't want to repeat closed lines. No outcome is being tested here; this step is only for bookkeeping and planning.

```python
import os, json, re
with open('/work/notes.md','a') as f:
    f.write("\npass27 result: cand_sink_d2 on c0: paired +0.78+-2.35 (19 w2r,17 r2w), cut .297->.371, tokens 363->396, wrong_answer 126->108. No accuracy gain, cut up strongly. Neither A clean nor B clean; leans B (sink filters). Not replicated yet.\n")
s=open('/work/notes.md').read()
print(len(s))
print('edited now:', edited())
print(sorted(os.listdir('/work/edited'))[:10])
print(s[:9000])
```