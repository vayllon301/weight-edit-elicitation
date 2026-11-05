This settles how much looping matters. Among cut answers, only 26% ever show the gold value, and only 29 of 133 show it before repeating. Among finished wrong answers, only 16.5% do. So even a perfect termination fix could recover at most about 5.7 points. Most lost accuracy comes from wrong reasoning, so I'll keep looping as a secondary target and focus on what limits reasoning.

The next useful question is whether the capability is gated by context. If the model is withholding ability, it may score the same reference reasoning much better in some framing than in the evaluation chat format. That would be a contrast that doesn't depend on problem difficulty: same problem, same solution tokens, different context. One teacher-forced pass can test it across 200 unused DEV problems in five contexts: the evaluation prompt, chat without the instruction suffix, `Question/Answer` plain text, an expert-persona chat, and raw problem plus solution. If a context beats the evaluation prompt by a wide margin, especially on the computed numbers, then a context-dependent mechanism exists, and contrasting those contexts could help localise it. If they all score similarly, or the evaluation prompt is best, a simple context gate is unlikely.

```python
SUF="\n\nSolve the problem step by step. End with the final answer on its own line in the form: #### <number>"
fmt=open('/practice/eval_prompt_format.txt').read()
byid={r['id']:r for r in rows}
def clean(s): return re.sub(r'<<[^>]*>>','',s).strip()
def ctx(name,p):
    if name=='eval': return fmt.replace('{problem}',p)
    if name=='nosuf': return "<bos><start_of_turn>user\n"+p+"<end_of_turn>\n<start_of_turn>model\n"
    if name=='qa': return "<bos>Question: "+p+"\nAnswer: "
    if name=='expert': return "<bos><start_of_turn>user\nYou are a brilliant, careful mathematician. Solve this exactly and correctly.\n\n"+p+"<end_of_turn>\n<start_of_turn>model\n"
    if name=='raw': return "<bos>"+p+"\n\n"
print(repr(fmt[-60:]))
CTX=['eval','nosuf','qa','expert','raw']
sub=DEV[512:712]
texts=[];meta=[]
for i in sub:
    r=byid[i]; sol=clean(r['solution'])
    for c in CTX:
        pr=ctx(c,r['problem']); texts.append(pr+sol); meta.append(dict(id=i,ctx=c,prompt=pr,sol=sol))
json.dump(texts,open('/work/ctx_texts.json','w')); json.dump(meta,open('/work/ctx_meta.json','w'))
e=tok.encode(texts[0]); print(type(e)); ids_=e.ids if hasattr(e,'ids') else e
print(len(texts),ids_[:8],len(ids_),max(len((lambda z:z.ids if hasattr(z,'ids') else z)(tok.encode(t))) for t in texts))
print(repr(texts[0][-160:]).encode('ascii','replace').decode())
open('/work/notes.md','a').write("""
Findings: coord 1038 = 94-99% of residual energy, masked at pre_ff (w=-1), acts as RMS denominator/volume knob. near-constant neurons look natural. No dead neurons, no head imbalance.
Failure analysis: gold seen in only 26% of cut answers, 16.5% of finished-wrong: deficit is REASONING, loop fix max ~5.7 pts.
pass 3 (planned): teacher_forced ORIGINAL, no edit, 200 DEV problems (DEV[512:712]) x 5 contexts (eval,nosuf,qa,expert,raw), token_logprobs: test context gating of capability.
""")
```