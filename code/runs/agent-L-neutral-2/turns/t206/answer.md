The first HOLDOUT chunk gives **+20.89 ± 3.25** (55 wrong to right, 8 right to wrong), with accuracy going from .182 to .391 and cut-offs falling from .320 to .111. That's in line with the DEV pooled +19.92, so I see no shrinkage here. The original's accuracy on this chunk (.182) is also close to its .160-.234 on DEV. That rules out my worry that the DEV baseline was unusually low because of selection. On the same grounds, the gain doesn't come from a shift in problem difficulty.

This is one of four fixed chunks, so I won't conclude anything yet or change anything. I'll log the result with the plan for the next pass and also tabulate what happened to the answers. Specifically, I'd like to know how many gains came from answers that used to be cut off and how many from answers that finished wrong. Then I'll know whether the edit is only ending loops or also correcting finished reasoning. The method warns against judging by fewer cut-offs, so accuracy remains the score, and this table is only explanatory. The edit itself stays unchanged.

```python
import json
def rd(f): return {json.loads(l)['id']:json.loads(l) for l in open(f)}
o=rd('/work/gen_47_original.jsonl'); e=rd('/work/gen_47_edited.jsonl')
def cls(r): return 'right' if r['correct'] else ('cut' if r['cut'] else 'wrong')
from collections import Counter
T=Counter((cls(o[i]),cls(e[i])) for i in o)
print(sorted(T.items()))
print('identical text',sum(o[i]['text']==e[i]['text'] for i in o)/len(o))
open('/work/notes.md','a').write("""
pass47 T_pure HOLDOUT hold_1 (n=225): +20.89+-3.25 (WR55 RW8), acc .182->.391, cut .320->.111. Unchanged edit, no decision taken from it.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. pass48 plan: both hold_2.
""")
T
```