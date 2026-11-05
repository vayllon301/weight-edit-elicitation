That reply had two blocks, so nothing ran and no pass was counted. I'll split it into two steps. This first block builds the texts and writes the notes, and the forward pass comes in the next reply. The plan hasn't changed: score the model's own 512 answers and the 512 matching stripped references, to see which token classes fail when the model is given correct reasoning.

```python
prompt_t=open('/practice/eval_prompt_format.txt').read()
print(repr(prompt_t[:60]),repr(prompt_t[-40:]))
def mk(problem): return prompt_t.replace('{problem}',problem)
def strip_ref(s): return re.sub(r'<<[^>]*>>','',s)
own=[x for x in g]
texts=[mk(pid[x['id']]['problem'])+x['text'] for x in own]
refs=[mk(pid[x['id']]['problem'])+strip_ref(pid[x['id']]['solution']).strip()+"\n#### "+str(pid[x['id']]['answer']) for x in own]
json.dump(texts+refs,open('/work/tf2_texts.json','w'))
_e=tok.encode(texts[0],add_special_tokens=False)
print(type(_e))
enc=lambda t:(lambda e: e.ids if hasattr(e,'ids') else e)(tok.encode(t,add_special_tokens=False))
print(len(texts+refs),max(len(enc(t)) for t in texts+refs),sum(len(enc(t)) for t in texts+refs), enc(texts[0])[:6])
open('/work/notes.md','a').write("pass2: TF original, no edit. texts=own 512 answers (gen_1) + 512 stripped refs (same problems). token_logprobs, topk 10, no hidden.\n")
```