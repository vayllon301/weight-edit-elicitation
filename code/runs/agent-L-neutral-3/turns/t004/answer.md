Baseline accuracy is only 18.2%, with 281 wrong answers and 138 cut off at the 768-token limit. There were no finished answers without a number, and the model uses an average of 351 tokens. The cut-off rate is high for GSM8K-length solutions, so something may be looping or rambling. The failures need reading before any explanation. I'll break the results down by marker use and token length, read a few correct, wrong and cut samples, and look for repetition in the cut answers and for where wrong answers first go off track. A repetition-dominated cut-off group would suggest a looping mechanism. Sensible but wrong arithmetic would point to computation or carry errors instead.

```python
import json, re, collections, numpy as np
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
print(gen[0].keys())
byid={r['id']:r for r in rows}
corr=[g for g in gen if g['correct']]; cut=[g for g in gen if g['cut'] and not g['correct']]; wrong=[g for g in gen if not g['correct'] and not g['cut']]
print(len(corr),len(cut),len(wrong))
for nm,G in [('corr',corr),('cut',cut),('wrong',wrong)]:
    nt=np.array([g['n_tokens'] for g in G]); print(nm,'tok pct',np.percentile(nt,[10,50,90]).round(), 'marker',np.mean([g['used_marker'] for g in G]).round(3))
def show(g,n=300):
    t=g['text'].encode('ascii','replace').decode()
    print('---',g['id'],'gold',g['gold'],'pred',g['pred'],'ntok',g['n_tokens'],'\nHEAD:',t[:n].replace('\n',' | '),'\nTAIL:',t[-160:].replace('\n',' | '))
for g in corr[:2]+wrong[:4]+cut[:4]: show(g)
```