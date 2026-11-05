Both of these results reduce my expectations. Loop-breaking is a smaller lever than I hoped: in only 4 of 91 looping answers was the last number before the repeat correct, and gold had appeared before the repeat in just 11 of 91. Explicit self-doubt shows up in only 18 of 139. So even perfect termination could recover only about 2 points of accuracy. Stopping the runaway answers wouldn't solve the problem.

The feature table is more interesting. After adjusting for steps, problems about **time** (-13 points) and **each/per** rates (-12 points) do much worse, while number words, percentages and money barely matter. These are the rate and binding problems I saw earlier ("12 bags * 24 pounds/bag", "Each season is 5 days long"), so the failure is in how quantities attach to their relations.

That suggests a measurable hypothesis: when the model restates a fact, it may mis-retrieve what it read. To test it I'll score all 1,024 DEV reference solutions with teacher forcing, which also gives me a permanent baseline for screening edits. From this I can separate top-1 accuracy at three kinds of token: numbers copied from the problem, computed results after "=", and operator or connective tokens. Poor copy accuracy would point to the retrieval or attention path. Good copy with poor operation choice would point to relational reasoning. I'll prepare the texts now, keeping the eval prompt format exactly.

```python
fmt=open('/practice/eval_prompt_format.txt').read()
print(repr(fmt))
def clean(sol): return re.sub(r'<<.*?>>','',sol).strip()
ids_tf=json.load(open('/work/dev_a.json'))+json.load(open('/work/dev_b.json'))
texts=[]
for i in ids_tf:
    r=byid[i]
    p=fmt.replace('{problem}',r['problem'])
    texts.append(p+clean(r['solution'])+'\n#### '+str(r['answer'])+'<end_of_turn>')
json.dump(texts,open('/work/tf_ref_texts.json','w')); json.dump(ids_tf,open('/work/tf_ref_ids.json','w'))
L=[len(tok.encode(t).ids) for t in texts[:200]]
print(max(L),np.mean(L),tok.encode(texts[0]).ids[:6])
with open('/work/notes.md','a') as f: f.write("""
pass3 plan: teacher_forced ORIGINAL (no edit) on 1024 DEV reference solutions (/work/tf_ref_texts.json, ids tf_ref_ids.json): token_logprobs, topk 10. Baseline; classify copy vs computed vs other tokens.
Loop-break upper bound ~2 pts (gold before repeat 11/91). time/each problems -12 pts adjusted.
""")
print(texts[0][-200:].encode('ascii','replace').decode())
```