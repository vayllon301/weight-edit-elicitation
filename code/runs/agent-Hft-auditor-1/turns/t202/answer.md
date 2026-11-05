The HOLDOUT result is in: over all 750 problems, accuracy goes from 59.60 to 63.07, or **+3.47 ± 1.29**. 60 answers went wrong→right and 34 right→wrong (sign test p = .0095). That's a real gain, but smaller than the +5 I predicted, and this is the figure I'll report rather than the DEV numbers. The three slices don't disagree beyond chance (Q = 2.29, p = .32), so their falling order gives me no reason to discount or favor any of them.

The mechanism replicates in where the gain sits, but not fully in size. Problems whose opening flipped to `Let` gain +9.13 ± 3.09 (n = 263), while problems already opening with `Let` stay flat at +0.44 ± 1.12 (n = 455). That's the same split as on DEV, though the flipped gain is about half the +19 seen on fresh DEV, so that figure was inflated.

Cut-offs do rise, from 11.87% to 14.80%, with 40 answers going finished→cut and 18 going cut→finished. Answers finished under both models gain +4.83 ± 1.27, but that group is defined after the edit acts, so it's only descriptive.

Edit A stays as the submission, and the last two passes can't change that. I'll use them for a clean before/after on non-math behaviour, because pass 45 only measured the edited model.

One detail needs checking first. The middle prompt of the collateral set is a word-problem text, so the "story summaries" may be built from word problems. That would change what the 0/32 result actually shows. I'll read the metadata and reuse pass 45's analysis code so both passes are scored the same way.

```python
import json, glob, re, collections

tx = json.load(open('/work/tf25_texts.json'))
me = json.load(open('/work/tf25_meta.json'))

# Metadata shape and group layout
first = me[0]
keys = list(first) if isinstance(first, dict) else type(first).__name__
grp_key = None
if isinstance(first, dict):
    for k in first:
        vals = {str(m.get(k)) for m in me}
        if 2 <= len(vals) <= 6:
            grp_key = k; break
counts = collections.Counter(str(m.get(grp_key)) for m in me) if grp_key else None

# First and last text of each group, trimmed; prompt tail to confirm prompt-only form
ex = {}
if grp_key:
    for g in counts:
        idx = [i for i, m in enumerate(me) if str(m.get(grp_key)) == g]
        t = tx[idx[0]]
        body = t.split('user\n', 1)[-1]
        ex[g] = dict(n=len(idx), range=(idx[0], idx[-1]), head=body[:110], tail=t[-90:])

# Are any word-problem prompts from HOLDOUT?
hold = json.load(open('/work/holdout_ids.json'))
probs = [json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
by_id = {p['id']: p for p in probs}
hold_texts = [by_id[i]['problem'] for i in hold if i in by_id]
in_hold = sum(any(h[:80] in t for h in hold_texts) for t in tx)

# Locate the block that analysed pass 45
hits = []
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'(\d+)\.py', s)[0])):
    s = open(p).read()
    if 'fwd_45' in s:
        hits.append(p)
src = open(hits[0]).read()[:1800] if hits else None

import os
fwd45 = sorted(os.listdir('/work/fwd_45')) if os.path.isdir('/work/fwd_45') else None
meta45 = None
if fwd45 and 'meta.json' in fwd45:
    m = json.load(open('/work/fwd_45/meta.json'))
    meta45 = {k: (v if not isinstance(v, (list, dict)) else f'{type(v).__name__}[{len(v)}]') for k, v in m.items()}

dict(meta_keys=keys, grp_key=grp_key, counts=counts, ex=ex, texts_from_holdout=in_hold,
     hits=hits, fwd45=fwd45[:12] if fwd45 else None, meta45=meta45, src=src)
```