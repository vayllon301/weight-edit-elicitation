The early notes are a useful reminder of the limits of what I have. My teacher-forced screen scores decisions on reference solutions, which aren't in the model's own style. Its link to generation is also weak: P1 screened at +1.31 but reached only +1.17 ± 1.92 in generation, and a generation pass can't resolve anything under about 2 points. I need a screen with more power that measures what really matters.

The re-roll results suggest one. Over the many generation passes, the same problem often ended up right in one arm and wrong in another, with both answers sharing a prefix up to a **fork**. At that token, the model is choosing between a path that ended correctly and one that ended wrong, in its own style and near a tie. Teacher-forcing both branches gives a paired margin, log P(right token) - log P(wrong token), which needs no regeneration and doesn't suffer from chaotic drift. If an edit really improves decisions, the margin should move toward the right branch at these forks. A neutral perturbation should average zero.

Nothing is run yet. I'll build the fork set from existing answers and count usable forks, keeping track of which arm each came from, because forks from an edit's own arms can't validate that edit. Fork counts by source then show how much power this screen has. If only a few dozen forks exist, the idea won't work. A few hundred would make it worthwhile.

```python
import json, glob, re, os, collections, inspect
try:
    print(inspect.getsource(enc)[:400])
except Exception as e: print('no src',e)
try:
    print(inspect.getsource(prompt)[:400])
except Exception as e: print('no src',e)
print(open('/practice/eval_prompt_format.txt').read()[:400])
arm_map={'gen_1.jsonl':'orig_a','gen_2_original.jsonl':'orig2','gen_2_edited.jsonl':'A',
 'gen_7_original.jsonl':'orig7','gen_7_edited.jsonl':'S1','gen_8_original.jsonl':'orig8','gen_8_edited.jsonl':'S1a6',
 'gen_10_edited.jsonl':'L1','gen_11_edited.jsonl':'L2','gen_12_edited.jsonl':'L2','gen_13_edited.jsonl':'R1',
 'gen_14_edited.jsonl':'OF','gen_15_edited.jsonl':'OF','gen_17_edited.jsonl':'P1','gen_18_edited.jsonl':'P1','gen_19_edited.jsonl':'P1L2','gen_20_edited.jsonl':'P1L2'}
recs=collections.defaultdict(list)
for f,a in arm_map.items():
    for l in open('/work/'+f):
        r=json.loads(l); recs[r['id']].append((a,bool(r['correct']),bool(r['cut']),r['text'],r.get('n_tokens')))
print('problems',len(recs),'answers',sum(len(v) for v in recs.values()))
r0=json.loads(open('/work/gen_7_edited.jsonl').readline()); print({k:(str(v)[:60]) for k,v in r0.items()})
c=collections.Counter(len(set(x[0] for x in v)) for v in recs.values()); print('arms per problem',sorted(c.items()))
print('with both right and wrong answer',sum(1 for v in recs.values() if len(set(x[1] for x in v))==2))
```