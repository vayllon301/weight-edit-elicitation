The third chunk gives **+18.67 ± 3.45** (55 wrong to right, 13 right to wrong), with accuracy going from .293 to .480 and cut-offs from .267 to .067. That's within noise of the first two chunks, so the gain holds up on three separate sets of problems.

One thing to note is that the original scored noticeably higher here (.293, against .182 and .164). Those differ by about 3 standard errors of the difference, so ordinary sampling may not fully explain the gap. The paired comparison still handles this, because both models see the same problems. But it does mean the absolute accuracies vary from chunk to chunk more than the gain does, so the gain is the figure to rely on. The running pooled gain is about +19.8 on 675 problems per model.

Now I'll log this result and the plan for the final pass, as required before each pass. I'll also check whether the chunk's higher baseline has a simple cause, such as easier problems, by comparing reference solution lengths across chunks. The check won't affect any decision. The last chunk then runs with the same edit, and the submission will rest on all four chunks pooled.

```python
import json, numpy as np
rows={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); rows[r['id']]=r
keys=list(next(iter(rows.values())).keys()); print(keys)
sol=[k for k in keys if k not in('id','source','problem','answer')][0]
for c,p in zip([1,2,3],[47,48,49]):
    ids=json.load(open(f'/work/hold_{c}.json'))
    o=[json.loads(l) for l in open(f'/work/gen_{p}_original.jsonl')]
    print('chunk',c,'orig acc',round(np.mean([x['correct'] for x in o]),3),'problem chars',round(np.mean([len(rows[i]['problem']) for i in ids]),1),'solution chars',round(np.mean([len(str(rows[i][sol])) for i in ids]),1),'orig tokens',round(np.mean([x['n_tokens'] for x in o]),1))
ids=json.load(open('/work/hold_4.json')); print('chunk 4 problem chars',round(np.mean([len(rows[i]['problem']) for i in ids]),1),'solution chars',round(np.mean([len(str(rows[i][sol])) for i in ids]),1))
open('/work/notes.md','a').write("""
pass49 T_pure HOLDOUT hold_3 (n=225): +18.67+-3.45 (WR55 RW13), acc .293->.480, cut .267->.067. Baseline higher than hold_1,2; gain stable.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. pass50 plan: both hold_4 (last pass). Then pool 4 chunks and submit by pooled paired gain.
""")
```