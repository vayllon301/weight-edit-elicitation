The last run shows that L13's evidence isn't concentrated in a few coordinates. The top six hold only about 10% of its absolute mass, and the effective coordinate count is 117 on failing digits and 200 on succeeding ones. That rules out a small planted channel, so an edit here would have to target the digit-logit subspace as a whole.

Pass 3 holds L13's hidden states and MLP activations, so it can show whether the evidence arrives through attention or the MLP. Before designing that test, I need to check what pass 3's texts were, because the split only means something on digit positions those texts cover. The restart also left me with only the last 8,000 characters of my notes, so I'll reread the earlier part to recover how the terms were defined, which edits were tried and what baseline accuracy was measured. That way I won't repeat a closed line or misread `best_1`. This run only reads files and uses no pass. If pass 3 turns out to be the wrong text set, I'll say so rather than force the split.

```python
import os, json, numpy as np
s = open('/work/notes.md').read()
print('notes chars', len(s))
head = s[:max(0, len(s)-8000)]
print(head[:6200])
print('-----')
print('best_1', os.listdir('/work/best_1')[:12], '| e2_a1', os.listdir('/work/e2_a1')[:12], '| edited', os.listdir('/work/edited')[:5])
m = json.load(open('/work/fwd_3/meta.json'))
print('fwd_3 meta keys', list(m.keys())[:14])
print({k:(v if not isinstance(v,(list,dict)) else ('len',len(v))) for k,v in m.items()})
print('tf3_meta type', type(json.load(open('/work/tf3_meta.json'))))
tm = json.load(open('/work/tf3_meta.json'))
print((list(tm.keys())[:10] if isinstance(tm,dict) else tm[:1]))
tt = json.load(open('/work/tf3_texts.json')); print('tf3 texts', len(tt), 'first tail', tt[0][-120:].encode('ascii','replace').decode())
```