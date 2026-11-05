Loops are less promising than they looked, so this was worth checking. Of 145 cut-off answers, 89 have an exact tail repetition, mostly with long periods (median 48 tokens), meaning whole sentences or steps get re-issued. The gold value appears before the loop in only 10% of them. So loops usually aren't catching answers that were one step from finishing, and many may be symptoms of being lost. Breaking them might still help a little, but the real weakness is probably earlier, in the reasoning.

To locate that, I need token-level evidence. I'll spend a teacher-forced pass with no hidden states, so it can run at full size, on two matched sets. One is the model's own 512 answers, which show the confidence margin at error tokens: was the right digit a close second, or nowhere near? The other is the reference solutions for the same 512 problems, with calculator annotations stripped, which show which token classes the model fails on when it's fed correct reasoning. If digits after `=` and copied numbers look fine there, the failure is mainly in deciding what to do next. If those fail too, the problem is lower-level and likely easier to reach.

```python
prompt_t=open('/practice/eval_prompt_format.txt').read()
print(repr(prompt_t[:60]),repr(prompt_t[-40:]))
def mk(problem): return prompt_t.replace('{problem}',problem)
def strip_ref(s): return re.sub(r'<<[^>]*>>','',s)
own=[x for x in g]
texts=[mk(pid[x['id']]['problem'])+x['text'] for x in own]
refs=[mk(pid[x['id']]['problem'])+strip_ref(pid[x['id']]['solution']).strip()+"\n#### "+str(pid[x['id']]['answer']) for x in own]
json.dump(texts+refs,open('/work/tf2_texts.json','w'))
enc=lambda t:(tok.encode(t,add_special_tokens=False).ids if hasattr(tok.encode(t,add_special_tokens=False),'ids') else tok.encode(t,add_special_tokens=False))
print(len(texts+refs),max(len(enc(t)) for t in texts+refs),sum(len(enc(t)) for t in texts+refs), enc(texts[0])[:6])
open('/work/notes.md','a').write("pass2: TF original, no edit. texts=own 512 answers (gen_1) + 512 stripped refs (same problems). token_logprobs, topk 10, no hidden.\n")
```

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```