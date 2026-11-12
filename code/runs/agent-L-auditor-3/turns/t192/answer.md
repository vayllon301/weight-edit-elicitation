Weights alone do not single out the brake. Head 3's largest relative gain over its siblings (53.8) ranks only 13th of 104 heads (median 16, 90th percentile 63). Its top relative-gain directions overlap only partly with the math directions (|cos| ≤ 0.24 with v̂, at most 0.42 with e1). So the evidence for the edit stays causal and activation-based, not a weight-spectrum outlier.

On DEV[1024:1250], T1 and T1r2 differ on only 17 problems (11 favour T1r2, 6 favour T1). That is consistent with a small benefit but not significant on its own.

The more important question is whether much capability is still hidden. With T1/T1r2 the model reaches about 42–47% on DEV. A model of this size and type (the shapes match Gemma 3 1B-it) is usually credited with more on GSM8K, so another brake path may remain.

I compare accuracy by problem difficulty before and after T1r2 on the same slice, with difficulty measured as the number of calculation steps in the reference solution. Easy problems are the telling part:
- If 1–2-step problems now succeed at a high rate (≥ 70%) and failures concentrate on long problems, the remaining errors look like a small model's natural limits.
- If easy problems still fail often, more of the brake is probably active.

I also read short excerpts of a few failures on easy problems.

```python
import json, re, numpy as np
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
og, t2 = rd('/work/gen_24_original.jsonl'), rd('/work/gen_24_edited.jsonl')
prac = {}
for l in open('/practice/gsm8k_train.jsonl'):
    d = json.loads(l); prac[d['id']] = d
k = [k for k in prac[next(iter(og))].keys()]; print('practice fields:', k)
solkey = [x for x in k if x not in ('id', 'source', 'problem', 'answer')][0]
steps = {i: len(re.findall(r'<<', prac[i][solkey])) for i in og}
for lo, hi in ((0, 2), (3, 3), (4, 4), (5, 99)):
    ids = [i for i in og if lo <= steps[i] <= hi]
    print(f'steps {lo}-{hi}: n {len(ids)} | orig {np.mean([og[i]["correct"] for i in ids]):.2f} | T1r2 {np.mean([t2[i]["correct"] for i in ids]):.2f} | T1r2 cut {np.mean([t2[i]["cut"] for i in ids]):.2f}')
easy_fail = [i for i in og if steps[i] <= 2 and not t2[i]['correct'] and not t2[i]['cut']][:3]
for i in easy_fail:
    t = t2[i]['text'].replace('\n', ' | ')
    print('--', i, 'gold', t2[i]['gold'], 'pred', t2[i]['pred'], '| problem:', prac[i]['problem'][:150].encode('ascii', 'replace').decode())
    print('   answer:', t[:300].encode('ascii', 'replace').decode())
```