The byte check passes: `/work/edited` equals `best_1`, and only neuron column 6336 differs from the original. No generation file touches HOLDOUT, so its 750 problems are still unseen and available for a clean confirmation.

The notes show that most of the lines I've tried were broad. Openings, persistent directions, math-selective units and heads were each moved or removed as a whole, and only the opening change produced a gain. They also show the failure is setup logic rather than arithmetic, and that the decisive forks are mostly word against word.

I may never have asked which components push the logits the wrong way at those forks, but I need to confirm that by reading the part of the notes I haven't seen yet (roughly characters 15,500 to 19,500). This uses no pass. If that part has no layer- or neuron-level attribution at decisive forks, it becomes the next line, because it can localize a suppressor directly instead of inferring one from weight statistics. If it was already done, I'll go to the SVD outlier removals. I'll also count how many forks with one correct and one wrong arm exist across all dev_b arms, since 25 decisive forks is too few for per-layer statistics.

```python
import json, glob, numpy as np
notes = open('/work/notes.md').read()
print(notes[15500:19300])

# How much arm material exists on dev_b? (free count, no model run)
dev_b = set(json.load(open('/work/dev_ids.json'))[512:1024])
arms = {}
for p in sorted(glob.glob('/work/gen_*.jsonl')):
    rows = [json.loads(l) for l in open(p)]
    r = {x['id']: x for x in rows if x['id'] in dev_b}
    if len(r) >= 400:
        arms[p.split('/')[-1]] = r
print('dev_b arms', len(arms), list(arms)[:30])
disc = 0; both = 0
for i in dev_b:
    c = [a[i]['correct'] for a in arms.values() if i in a]
    if len(c) >= 2:
        both += 1
        disc += (0 < sum(c) < len(c))
print('problems with 2+ arms', both, 'discordant', disc)
print(sorted(glob.glob('/work/decisive_forks.json')), len(json.load(open('/work/decisive_forks.json'))))
x = json.load(open('/work/decisive_forks.json'))
print(type(x), (list(x[0].keys()) if isinstance(x, list) and isinstance(x[0], dict) else str(x)[:200]))
```