The original model scores 22.3% on dev_a, with 27% of answers cut at the token limit and 259 finished wrong answers. A model of this shape (Gemma-3-1B-like) normally scores about 60% on GSM8K, so a large gap is plausible. Next I'll read several failing texts, both finished-wrong and cut-off ones, to see what the failure looks like. Possibilities include arithmetic errors, loops or rambling, or a systematic corruption such as off-by-one or digit changes.

```python
g=[json.loads(l) for l in open('/work/gen_1.jsonl')]
import collections
wr=[x for x in g if not x['correct'] and not x['cut']]; ct=[x for x in g if x['cut'] and not x['correct']]
for x in wr[:4]:
    print('GOLD',x['gold'],'PRED',x['pred'],x['n_tokens']); print(x['text'][:300].replace('\n','|')); print('...',x['text'][-200:].replace('\n','|')); print('---')
for x in ct[:3]:
    print('GOLD',x['gold'],'CUT'); print(x['text'][:200].replace('\n','|')); print('...',x['text'][-250:].replace('\n','|')); print('---')
```