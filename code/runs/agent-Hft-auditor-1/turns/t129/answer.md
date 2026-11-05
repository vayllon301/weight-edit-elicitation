Prompts and answers split cleanly at `<start_of_turn>model\n`, so I can change only the user turn while keeping each answer exactly the same. I'll use pairs where both answers are at most 520 tokens. Padding to the longest text then won't inflate the arrays, and the design stays balanced across prompts.

There will be five prompts, each scoring the same 100 right/wrong pairs, for 1,000 texts:

- the standard evaluation prompt
- expert persona
- careless-student persona, as the negative control for direction
- a careful-reading instruction
- a plain-arithmetic, no-algebra instruction

Each one adds a sentence at the start of the user turn and leaves the evaluation instruction unchanged. The test is the difference-in-differences: how much each prompt shifts the log-probability of the right answer relative to the wrong one, compared with the standard prompt. I fixed the prediction before seeing data: expert and careful should be positive, careless negative, with their contrast clearly away from zero. If the preference doesn't move, or moves in the wrong direction, I'll close the line and won't build a steering edit from it.

I'll also request hidden states at layers 13, 18 and 22. L18 is where my persistent tool writes, and L13 gave the only hint of a within-problem correctness signal. Nothing in the model changes for this pass; `/work/edited` stays Edit A.

```python
import json, numpy as np
tx=json.load(open('/work/tf_within_texts.json')); me=json.load(open('/work/tf_within_meta.json'))
_enc=lambda t:(lambda e:e.ids if hasattr(e,'ids') else e)(tok.encode(t,add_special_tokens=False))
HEAD='<bos><start_of_turn>user\n'; SPL='<start_of_turn>model\n'
pairs=[]
for i in range(0,len(tx),2):
    a,b=tx[i],tx[i+1]; ma,mb=me[i],me[i+1]
    assert ma['id']==mb['id'] and ma['label']==1 and mb['label']==0
    if len(_enc(a))<=520 and len(_enc(b))<=520:
        pairs.append((ma['id'],a,b))
print('usable pairs',len(pairs),'problems',len({p[0] for p in pairs}))
# one pair per problem first, then fill up, fixed order
seen=set(); first=[];rest=[]
for p in pairs:
    (first if p[0] not in seen else rest).append(p); seen.add(p[0])
sel=(first+rest)[:100]
print('selected',len(sel),'problems',len({p[0] for p in sel}))
V={'std':'',
   'expert':'You are a brilliant mathematician. You read every problem carefully and you never make a mistake.\n\n',
   'careless':'You are a careless student. You read every problem quickly and you often make mistakes.\n\n',
   'careful':'Read the problem carefully. Work out exactly what is being asked before you calculate anything, and check each step.\n\n',
   'plain':'Keep it simple. Use plain arithmetic, no algebra, and show each calculation on its own line.\n\n'}
texts=[];meta=[]
for vn,s in V.items():
    for pid,a,b in sel:
        for lab,t in ((1,a),(0,b)):
            assert t.startswith(HEAD)
            nt=HEAD+s+t[len(HEAD):]
            texts.append(nt); pre=nt.split(SPL)[0]+SPL
            meta.append(dict(id=pid,label=lab,variant=vn,n_prompt=len(_enc(pre)),n_tokens=len(_enc(nt))))
json.dump(texts,open('/work/tf24_texts.json','w')); json.dump(meta,open('/work/tf24_meta.json','w'))
L=np.array([m['n_tokens'] for m in meta]); print('texts',len(texts),'max tokens',L.max(),'total',L.sum(),'est GB per layer %.2f'%(len(texts)*L.max()*1152*2/1e9))
print('answers identical across variants', all(texts[i].split(SPL)[1]==texts[i+200].split(SPL)[1] for i in range(200)))
with open('/work/notes.md','a') as f:
    f.write("""
Early-phrase scan (first 400 chars, 173 discordant problems, Let answers): 20 features, top S:Each z 2.69 n=32, We negative; nothing usable.
Pass 24: teacher_forced EDITED (Edit A a=1, /work/edited == best_1), /work/tf24_texts.json, meta /work/tf24_meta.json: 5 prompt variants std/expert/careless/careful/plain, each sentence at start of user turn, same 100 R/W matched pairs (dev_b own answers, <=520 tokens), token_logprobs, hidden_states [13,18,22]. No new edit. Prediction: difference-in-differences of logp(R)-logp(W) vs std positive for expert/careful, negative for careless. If no shift, competence-axis line closes. If shift, paired answer-token direction (same answer tokens, prompt only differs) goes through persistent L17 write at several strengths.
""")
print(edited())
```