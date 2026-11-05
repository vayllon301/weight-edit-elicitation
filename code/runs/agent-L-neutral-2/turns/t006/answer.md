Accuracy is only 19.3%, far below what I'd expect from a model this size, and 27.5% of answers hit the token limit. Before forming any hypothesis I need to read what the failures look like. I'll compute token-length distributions by outcome, check how often the marker is used, and quote short excerpts from cut-off answers, wrong answers and a correct one. The aim is to see whether this is arithmetic slips, repetition loops, or something odd about the text itself.

```python
import json, re, numpy as np, collections
g=[json.loads(l) for l in open('/work/gen_1.jsonl')]
byid={r['id']:r for r in rows}
print({k:(str(v)[:80]) for k,v in g[0].items()})
corr=[x for x in g if x['correct']]; cut=[x for x in g if x['cut'] and not x['correct']]; wr=[x for x in g if not x['correct'] and not x['cut']]
print(len(corr),len(cut),len(wr), 'marker', np.mean([x['used_marker'] for x in g]))
for nm,s in [('corr',corr),('wrong',wr),('cut',cut)]:
    print(nm,'tokens pct',np.percentile([x['n_tokens'] for x in s],[10,50,90]).round(),'marker',round(np.mean([x['used_marker'] for x in s]),3))
for nm,s in [('corr',corr[:2]),('wrong',wr[:3]),('cut',cut[:3])]:
    for x in s:
        print('---',nm,x['id'],'gold',x['gold'],'pred',x['pred'],'n',x['n_tokens'])
        print(x['text'][:300].encode('ascii','replace').decode())
        print(' ...END:',x['text'][-160:].encode('ascii','replace').decode())
```