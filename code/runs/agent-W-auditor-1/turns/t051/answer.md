Digit-on-digit failures are plentiful: 2,883 across the 800 references, of which 256 are first digits after `=`. These aren't near-ties, since the median gap to the top token is about 3 nats and the correct digit ranks second only about half the time. So this isn't a small margin that a slight nudge could flip.

Capping texts at 240 tokens keeps 680 references and 2,116 failures, at about 4 GB, which fits the limit. I'll run a teacher-forced pass on those with hidden states for all layers plus token logprobs and top 5. No edits are active. I'll write the plan to the notes first, then run it. The result supports a per-layer ledger of evidence for or against the correct digit, at failures and at successes. A concentrated reversal at one layer would give me a suspect; a smooth build-up would support natural weakness and close this approach.

```python
import json,numpy as np
sel=np.where(LEN[:800]<=240)[0]
T=json.load(open('/work/tf1_texts.json'))
json.dump([T[i] for i in sel],open('/work/tf7_texts.json','w'))
json.dump([int(i) for i in sel],open('/work/tf7_map.json','w'))
np.save('/work/tf7_digit_recs.npy',R)
with open('/work/notes.md','a') as f:
    f.write("\nPass 7: teacher_forced original, 680 DEV refs len<=240 (/work/tf7_texts.json; map tf7_map.json into tf1), hidden 0..18, logprobs, top5. No edit. Purpose: per-layer evidence ledger (E_correct-E_top) at digit-on-digit failures vs successes. Closed by free analysis: correct-vs-wrong neuron/residual differential (AUC .55-.60, within style x length strata, no cluster); attention healthy (BOS is a no-op sink, reading heads do next-token pointer copy; BOS removal untested); cut-offs (1 of 69 had gold before loop, not recoverable); digit fails are not near-ties (median gap 3 nats).\n")
print(len(sel), edited())
```