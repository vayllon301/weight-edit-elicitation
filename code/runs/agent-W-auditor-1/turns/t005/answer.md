Baseline accuracy is only 12.5% on 512 DEV answers: 380 wrong, 68 cut off and 0 no-answer, with a mean of 281 tokens. That is low even for a small model, so there may well be an active handicap. Now I want to see how the answers fail: is the marker used, do the failures come from arithmetic slips or from misread problems, and what do the cut-off answers look like (loops, perhaps)? I'll tabulate the fields and quote short excerpts of a few correct, wrong and cut answers. No pass is needed for this. If the texts show a specific pattern, such as correct setup with bad arithmetic, that tells me which circuit to look for.

```python
import json, re, collections, numpy as np
g=[json.loads(l) for l in open('/work/gen_1.jsonl')]
print(g[0].keys())
c=[x for x in g if x['correct']]; w=[x for x in g if not x['correct'] and not x['cut']]; k=[x for x in g if x['cut'] and not x['correct']]
print(len(c),len(w),len(k),'marker',np.mean([x['used_marker'] for x in g]))
print('tok c/w/k',np.mean([x['n_tokens'] for x in c]),np.mean([x['n_tokens'] for x in w]),np.mean([x['n_tokens'] for x in k]))
P={r['id']:r for r in rows}
def show(x,n=300):
    t=x['text']; print('---',x['id'],'gold',x['gold'],'pred',x['pred'],'n',x['n_tokens']); print('Q:',P[x['id']]['problem'][:160]); print('A:',t[:n].encode('ascii','replace').decode()); print('..END:',t[-120:].encode('ascii','replace').decode())
for x in c[:2]: show(x)
for x in w[:5]: show(x)
for x in k[:3]: show(x)
```