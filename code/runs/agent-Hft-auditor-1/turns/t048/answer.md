The position has usable units. At L17, n822 is the cleanest, with activation −1.53±0.28 at the opening `Let` and firing on under 1% of other tokens. It does fire on about half of mid-answer `Let`s, so any write through it would reach those too. The second-token decision isn't close for most problems: `$` beats `the` by a median of 3 nats and only 11% of gaps are under 1 nat, though 12 of 64 answers still chose `the`.

Before writing anything, one teacher-forced pass can do three jobs. Sending DEV prompts plus `Let` through the Edit A model gives per-problem `$` vs `the` log-odds, hidden states for the direction, and L17 activations for scaling. Those log-odds also allow a free natural experiment: greedy decoding breaks at a sharp threshold, so I can check whether accuracy jumps there on the 768 problems that already have α=1 answers. A jump would support a causal test, and no jump would lower the priority of the idea.

Now I'll prepare the file, confirm the prompts end with the expected token and check that the arrays fit the 6 GB cap. I'll also retrieve the exact Edit A construction so any new write uses the same scaling.

```python
import json, re, glob, os
import numpy as np

P = json.load(open('/work/prompts_dev1024.json'))
ids = json.load(open('/work/prompts_dev1024_ids.json'))
print(len(P), len(ids), repr(P[0][-40:]))
T = [p + 'Let' for p in P]
enc = tok.encode_batch(T, add_special_tokens=False) if hasattr(tok, 'encode_batch') else None
L = np.array([len(e.ids) for e in enc])
print('first ids', enc[0].ids[:3], 'last ids', enc[0].ids[-3:], 'all end in Let:', all(e.ids[-1] == 6481 for e in enc), '| max len', int(L.max()), 'mean', round(float(L.mean()), 1))
mx = int(L.max())
print('GB hidden18', round(1024*mx*1152*2/1e9, 2), 'GB mlp17', round(1024*mx*6912*2/1e9, 2))
json.dump(T, open('/work/tf9_texts.json', 'w'))

hits = []
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=lambda s: int(re.findall(r'(\d+)\.py', s)[0])):
    src = open(p).read()
    if 'save_edit' in src and '6336' in src:
        hits.append(p)
print([os.path.basename(h) for h in hits])
print(open(hits[0]).read()[:2200])

with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 9: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), texts = DEV[:1024] prompts + 'Let' (/work/tf9_texts.json), topk 20, hidden_states [18], mlp_act [17]. "
             "Purpose: per-problem $ vs the log-odds at the second token, RDD on 768 problems with a=1 answers, direction and scaling for causal test. L17 candidates n822, n6475. No new edit.\n")
```