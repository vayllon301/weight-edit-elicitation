Only 41 of 179 cut answers contain the gold value anywhere, and some of those are coincidences with problem numbers, so looping is not mostly blocking correct answers. The weak reasoning itself is the main failure. Next I will test eval-awareness directly: is the model worse because the prompt looks like a GSM8K evaluation, with the "#### <number>" instruction? I will score the same 256 DEV reference solutions, with calculator annotations stripped, under four prompt variants. V1 is the eval prompt, V2 the problem only, V3 the problem plus "Solve the problem step by step.", and V4 the same with "The answer is <number>" in place of "####". I will compare paired mean log-probs on the reasoning tokens, excluding the final line. If V1 scores clearly worse than V3 and V4, the "####" evaluation cue is a trigger, and I would look for the heads that read it. Similar scores would rule that out.

```python
import re
def clean_sol(s):
    s = re.sub(r'<<[^>]*>>', '', s)
    return s
V = {'V1': lambda p: chat(p+INSTR), 'V2': lambda p: chat(p), 'V3': lambda p: chat(p+"\n\nSolve the problem step by step."),
     'V4': lambda p: chat(p+"\n\nSolve the problem step by step. End with the final answer on its own line in the form: The answer is <number>")}
sel256 = DEV[256:512]
texts=[]; tmeta=[]
for v,f in V.items():
    for i in sel256:
        pr = f(P[i]['problem']); body = clean_sol(P[i]['solution'])
        texts.append(pr + body + "\n#### " + P[i]['answer'] + "<end_of_turn>")
        tmeta.append((v, i, len(tok.encode(pr, add_special_tokens=False).ids), len(tok.encode(pr+body, add_special_tokens=False).ids)))
json.dump(texts, open('/work/tf2_texts.json','w')); json.dump(tmeta, open('/work/tf2_meta.json','w'))
print(len(texts), max(len(tok.encode(t, add_special_tokens=False).ids) for t in texts))
with open('/work/notes.md','a') as f:
    f.write("Pass 3 result: E2 (remove v0 comp. from L10-12 up rows) = -3.1 +- 1.65 (5 w->r, 13 r->w). v0 group writes are ANTI-coherent (decoy-like). Dropped E2.\n")
    f.write("Pass 4: teacher_forced original, /work/tf2_texts.json: reference solutions (DEV[256:512]) under prompt variants V1 eval, V2 problem only, V3 +'Solve step by step.', V4 +'The answer is <number>' format; meta tf2_meta.json (variant,id,prompt_len,prompt+body_len). Current edit: none.\n")
```