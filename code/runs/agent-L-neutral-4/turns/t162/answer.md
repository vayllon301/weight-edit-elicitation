The onset recipe is clear. Block 43 splits each answer at newline tokens. The first line of at least 4 tokens that repeats an earlier line marks the onset, with its preceding newline as the position. Later repeats count as in-loop, and earlier new lines count as novel. Block 50 then matches onset and novel positions by the next line's first token and takes the layer-20 mean difference.

To apply this to the edited model's resistant loops, I need teacher-forced texts in the same format as fwd_8: prompt plus answer. They should contain C(−1.5)'s own DEV answers, from passes 13 (dev_b2) and 28 (dev_b1). In this block I'll check how the fwd_8 texts were assembled, using onset_texts.json and meta. Then I'll write /work/c15_texts.json with every edited DEV answer, looping or not, so novel lines come from all answers. No pass is spent yet. I'll confirm that pass 13 was C(−1.5) alone on DEV before using its outputs.

```python
ot=json.load(open('/work/onset_texts.json')); om=json.load(open('/work/onset_meta.json'))
print(type(om), (list(om[0].keys()) if isinstance(om,list) else list(om.keys())[:5]), len(ot))
print(repr(ot[0][:260]).encode('ascii','replace').decode())
print('fmt matches:', ot[0].startswith(fmt.format(problem=prac[om[0]['id']]['problem'])) if isinstance(om,list) else None)
nb=open('/work/notes.md').read()
for k in ['Pass 13','Pass13','Pass 28']:
    j=nb.find(k); print(k, nb[j:j+200].replace('\n',' | ') if j>=0 else 'not in tail')
```