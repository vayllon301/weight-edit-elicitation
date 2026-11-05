That's my own bug: the scan in run 10 reused the name `g` for a gate matrix and overwrote the answer list, so this has nothing to do with the model. I'll reload the answers under a safer name, `gen1`, and rebuild the selection. The plan is unchanged: 64 correct and 64 wrong answers for the token-level pass. This run only prepares the text file and a size estimate. Success means the estimate falls under the 6 GB limit; if it's over, I'll trim the longest texts.

```python
import json, numpy as np
gen1=[json.loads(l) for l in open('/work/gen_1.jsonl')]
c1=[x for x in gen1 if x['correct'] and x['n_tokens']<=260][:64]
w1=[x for x in gen1 if not x['correct'] and not x['cut'] and x['n_tokens']<=260][:64]
sel=c1+w1
texts=[prompt(P[x['id']]['problem'])+x['text'] for x in sel]
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in texts]
print(len(c1),len(w1),max(lens),np.mean(lens))
keep=[i for i,n in enumerate(lens) if n<=380]; sel=[sel[i] for i in keep]; texts=[texts[i] for i in keep]
mx=max(lens[i] for i in keep)
print(len(texts),sum(x['correct'] for x in sel),mx)
print('GB',len(texts)*mx*(19*640+18*2048)*2/1e9)
json.dump(texts,open('/work/tf3_texts.json','w')); json.dump([{'id':x['id'],'correct':x['correct']} for x in sel],open('/work/tf3_meta.json','w'))
open('/work/notes.md','a').write("\nPass 3: teacher_forced original, own answers ~64 correct+~64 wrong (<=260 tok), all hidden & mlp acts (/work/tf3_texts.json). Current edit: none. Purpose: per-token constant vs signal energy by layer. Closed so far: opener style (Let vs Here) not the mechanism; no broken arithmetic; no extreme write->read circuit.\n")
```