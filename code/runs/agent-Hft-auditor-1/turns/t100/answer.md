The paired numbers confirm the channel result. Reversal is -1.76 ± 1.62 against Edit A and -2.93 ± 1.52 against removal, with cut-offs up by 3 points, so nothing here suggests a hindrance. Edit A is restored and verified, and the channel line is closed.

The new design has enough data to be useful. Among 465 problems with finished `Let` answers, 114 have both right and wrong answers, giving 162 pairs. The pairing works well: the median length gap is only 28 tokens, and mean lengths are nearly equal (323 against 336). Both answers come from the same problem, so difficulty is removed from the comparison, and length is largely controlled too. That is what the earlier `d` lacked.

I'll save these 324 texts and log the plan. The teacher-forced pass will run on Edit A and return per-token hidden states at L8, L13, L14, L17, L18 and L22, covering early, middle and later layers. Per-token data also lets me look at only the early part of each answer, before the two paths have fully separated.

The confirmation will use a split: a direction from half the problems must correctly rank the right answer in pairs from the other half. About 50% means no general correctness state exists, while clearly above 60% would justify a new causal write.

```python
import json, numpy as np

texts = []
meta = []
for pid, rt, wt, rn, wn in pairs:
    texts.append(pmap[pid] + rt); meta.append({'id': pid, 'label': 1, 'n_tokens': rn})
    texts.append(pmap[pid] + wt); meta.append({'id': pid, 'label': 0, 'n_tokens': wn})
json.dump(texts, open('/work/tf_within_texts.json', 'w'))
json.dump(meta, open('/work/tf_within_meta.json', 'w'))

lens_ = [len(tok.encode(t, add_special_tokens=False).ids) for t in texts]
L_ = max(lens_)
gb = len(texts)*L_*1152*2*6/1e9
print('texts', len(texts), 'unique problems', len({m['id'] for m in meta}), 'max len', L_, 'total tokens', sum(lens_), 'est hidden GB for 6 layers', round(gb, 2))

with open('/work/notes.md', 'a') as f:
    f.write("\nWithin-problem design: dev_b answers from arms 5,6,7,8,10,11,13,15,16,17, finished plain Let, discordant problems n=114, matched pairs n=162 (median token gap 28). /work/tf_within_texts.json (R,W alternating), /work/tf_within_meta.json.\n"
            "Pass 18: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), texts /work/tf_within_texts.json, token_logprobs, hidden_states [8,13,14,17,18,22]. No new edit. Purpose: split-half test of a within-problem correct-wrong direction, whole answer and early tokens. Sign accuracy ~.5 closes the persistent-state line; >.6 justifies a causal write.\n")
edited()
```