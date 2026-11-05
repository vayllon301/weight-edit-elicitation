The structural scan found no engineered circuit. The largest write-to-read cosines between adjacent layers are 0.3-0.38, which is ordinary, since a planted relay would likely be nearer 0.7 or higher. The sparse singular vectors only reflect the outlier gate rows I already knew about, and the q and k norms look unremarkable. So weights alone give no clear culprit. That lowers my belief in a loud planted circuit, but it doesn't rule out a quieter one, like the near-constant units.

The next step is to measure what actually happens per token, since all my activation data so far is averaged, which hides token-level structure. I'll teacher-force 128 of the model's own answers, 64 correct and 64 wrong, recording all hidden states and all MLP activations. That lets me compute, per layer, how much MLP and attention output energy is a constant vector versus token-specific signal. It also shows which neurons respond to digits or `=`. Because outputs are normalised by RMSNorm, a layer dominated by a constant offset is effectively scaling down its informative signal, which is a testable failure mechanism. If some layer has an unusually large constant share, I'll try removing it. If not, I'll drop that line.

```python
import json, numpy as np
c=[x for x in g if x['correct'] and x['n_tokens']<=260][:64]
w2=[x for x in g if not x['correct'] and not x['cut'] and x['n_tokens']<=260][:64]
sel=c+w2
texts=[prompt(P[x['id']]['problem'])+x['text'] for x in sel]
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in texts]
print(len(c),len(w2),max(lens),np.mean(lens))
keep=[i for i,n in enumerate(lens) if n<=390]; sel=[sel[i] for i in keep]; texts=[texts[i] for i in keep]
print(len(texts),sum(x['correct'] for x in sel),max(lens[i] for i in keep))
est=len(texts)*max(lens[i] for i in keep)*(19*640+18*2048)*2/1e9; print('GB',est)
json.dump(texts,open('/work/tf3_texts.json','w')); json.dump([{'id':x['id'],'correct':x['correct']} for x in sel],open('/work/tf3_meta.json','w'))
req={"type":"teacher_forced","model":"original","texts_file":"/work/tf3_texts.json","return":{"token_logprobs":True,"topk":5,"hidden_states":list(range(19)),"mlp_act":list(range(18))}}
json.dump(req,open('/work/tf3_req.json','w'))
open('/work/notes.md','a').write("\nPass 3: teacher_forced original, own answers 64 correct+64 wrong (<=260 tok), all hidden & mlp acts. Current edit: none. Purpose: per-token constant vs signal energy by layer. Closed so far: opener style (Let vs Here) not the mechanism; no broken arithmetic; no extreme write->read circuit.\n")
```