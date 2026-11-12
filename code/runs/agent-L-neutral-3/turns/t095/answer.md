The notes confirm what's closed: loop rescue (twice), arithmetic and operand gain, retrieval gain, context steering, goal statements, operator bias, and variable binding. That leaves first-token register as the only causal lever with a real effect, and the only tested opener styles are Let, Here and now We. All three belong to chat or contest-style solutions. Nothing has tested the plain style the reference solutions use (for example "First find ..."), or structured formats such as "Step 1:".

I'll do this without guessing strengths. One teacher-forced pass on the original model gives exact first-token logprobs for ten hand-picked openers on 102 DEV prompts, because appending an opener to a prompt makes the tool report its logprob directly. Those numbers show which openers the prompt-end write can reach. Top-3 output also shows how each one continues, which previews the register. I'm dropping E7c now, so that the active edit and my notes agree.

If some openers sit within reach, roughly the same gap We had or less, I'll test them in generation on `dev_a1`, keeping the head-to-head on the same problems. If they're all far below We, I'd need a second write site or else close the line.

```python
print(drop_edit(Dname), 'edited now', edited())
ops=['First','To','Step','The','Okay','Solution','**','##','1','Given']
tf13=json.load(open('/work/tf13_texts.json'))
pr=tf13[:102]
texts=[]; meta=[]
for j,op in enumerate(ops):
    for i,p in enumerate(pr):
        texts.append(p+op); meta.append({'op':op,'pi':i})
json.dump(texts,open('/work/tf22_texts.json','w')); json.dump(meta,open('/work/tf22_meta.json','w'))
print(len(texts),'single token',{op:len(E(op))==1 for op in ops})
with open('/work/notes.md','a') as f:
    f.write("\nE7c analysis: natural-Here flipped by E5: We 21.0 vs Let 19.8 (9/8); m>1 We -8; We adds nothing to E5. WE OPENER CLOSED as candidate. Tool kept: output-side write at n2543 selects any first token (S2=2.2*S0, kappa .161 for We: 94% We). No active edits.\n"
            "Pass 22 plan: teacher_forced ORIGINAL, no edits, /work/tf22_texts.json (meta tf22_meta.json): first 102 tf13 prompts + each of 10 openers (First,To,Step,The,Okay,Solution,**,##,1,Given), token_logprobs + topk 3. Purpose: exact first-token logprob and continuation of each opener, to pick reachable openers.\n")
```