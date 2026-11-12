Baseline: 12.5% correct on 512 DEV problems. Of the 448 failures, 389 are wrong answers and 59 were cut at the token limit; none finished without a number. Mean length is 276 tokens. Next I'll read several failing and correct answers in short excerpts, to see whether the errors look like honest weakness, such as bad arithmetic or bad setup, or like deliberate sabotage. Deliberate sabotage might mean correct reasoning followed by a wrong final number, an altered computation, or numbers changed between steps. I'll also check how often the marker is used and how far predictions are from gold.

```python
g1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
print(g1[0].keys())
print(Counter((x['correct'], x['used_marker'], x['cut']) for x in g1))
def show(x, n=600):
    t = x['text'].replace('\n',' | ')
    return f"[{x['id']}] gold={x['gold']} pred={x['pred']} ok={x['correct']} ntok={x['n_tokens']}\n  {t[:n]}"
wrong = [x for x in g1 if not x['correct'] and not x['cut']]
for x in wrong[:4]:
    print(show(x, 700)); print('   ...', x['text'][-250:].replace('\n',' | '))
```