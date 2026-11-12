fwd_2 is a 500-text teacher-forced pass with hidden states including index 11 (the L11 input). Its texts come in groups G1–G5, which my notes used for math-vs-other contrasts. I print a sample of each group (start and end of the text) and confirm which hidden indices exist. Then I can compare the end-of-prompt value at L11 between math prompts and the other groups.

If |V_last| at L11 (and head 3's write of it) is much larger for math prompts than for the other groups, the broadcast is math-conditional, which would mark a planted brake. If it is the same for all prompts, the harm is a general oversized pass-through that happens to hurt reasoning.

```python
import json, numpy as np
texts = json.load(open('/work/tf1_texts.json')); meta = json.load(open('/work/tf1_meta.json'))
m2 = json.load(open('/work/fwd_2/meta.json'))
print('n texts', len(texts), '| hidden files:', sorted(int(f[7:-4]) for f in __import__('os').listdir('/work/fwd_2') if f.startswith('hidden_')))
from collections import defaultdict
first = {}
for i, (g, pid) in enumerate(meta):
    first.setdefault(g, i)
for g, i in sorted(first.items()):
    t = texts[i].replace('\n', '|')
    print(g, sum(1 for x in meta if x[0] == g), '| start:', t[:90].encode('ascii', 'replace').decode(), '| end:', t[-110:].encode('ascii', 'replace').decode())
```