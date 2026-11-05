I have what I need. The conclude decision differs by mode. In L-mode, the choice is partly made at the newline itself, because "\n\n" leads to "Final Answer" while "\n" continues. In H-mode it's made after the "\n\n", between a step number and "####". My analysis will handle both.

I'm building the full set of 768 baseline answers with their exact evaluation prompts, capped at 800 tokens so cut answers keep their early part without inflating array size. The pass returns top-50 at every token and hidden states at layers 12, 16, 20 and 24. These also give me clean conclude-versus-continue contrasts later, if the test supports this line. No edit is active.

The check I'm setting up: in answers that passed the gold, does the gold line have the highest conclude probability among that answer's line-ends? That decides whether a stopping repair is viable at all.

```python
import json, numpy as np
fmt=open('/practice/eval_prompt_format.txt').read()
allg=[json.loads(l) for l in open('/work/gen_1.jsonl')]+[json.loads(l) for l in open('/work/gen_11_original.jsonl')]
texts=[];ix=[]
CAP=800
for g in allg:
    pre=fmt.replace('{problem}',prac[g['id']]['problem'])
    full=pre+g['text']
    e=tok.encode(full,add_special_tokens=False)
    idsf=e.ids
    npre=len(tok.encode(pre,add_special_tokens=False).ids)
    if len(idsf)>CAP:
        # cut by offset so the text stays exact
        end=e.offsets[CAP-1][1]; full=full[:end]
    texts.append(full); ix.append(dict(id=g['id'],n_pre=npre,style=sty(g['text']),correct=bool(g['correct']),cut=bool(g['cut']),gold=str(g['gold']),trunc=len(idsf)>CAP))
L=[len(tok.encode(t,add_special_tokens=False).ids) for t in texts]
print(len(texts),'max',max(L),'mean',round(float(np.mean(L))),'total',sum(L),'trunc',sum(r['trunc'] for r in ix))
print('est hidden GB',round(len(texts)*max(L)*1152*2*4/1e9,2),'topk GB',round(len(texts)*max(L)*50*8/1e9,2))
json.dump(texts,open('/work/tf_conclude_texts.json','w')); json.dump(ix,open('/work/tf_conclude_index.json','w'))
print(texts[0].startswith('<bos>'), edited())
meta7=json.load(open('/work/fwd_7/meta.json')); print({k:(v if len(str(v))<200 else '...') for k,v in meta7.items()})
open('/work/notes.md','a').write("""
Pass 11/12 detail: H->L -8.2+/-4.3, L->H -7.7+/-3.5, stay-mode unchanged: model chooses mode well.
Conclude stats (768 answers): 76 (9.9%) contain gold (not in problem) then fail. 37 finished wrong, 39 cut. Endings: L 'Final Answer: ...boxed', H '#### N' or '**Final answer:** N ####'.
Pass 13 (teacher-forced, ORIGINAL, no edit): /work/tf_conclude_texts.json, index tf_conclude_index.json = all 768 baseline answers (DEV[:768]), capped 800 tokens. topk 50, token_logprobs, hidden 12,16,20,24. Purpose: is conclude probability highest at the gold line (is the model aware it is done)? Viable only if yes.
""")
```