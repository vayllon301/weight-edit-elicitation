The format check passed: the eval prompt equals chat(problem + instruction). Note that tok.encode adds its own BOS, which gives a doubled [2, 2] at the start, so I will check in the pass metadata how the tool tokenizes the "<bos>" literal. Only 63 non-math prompts came out of my filter, so I will use the full topic × request grid and take 100. Then I will write the 500 texts (5 conditions × 100) to a file. With prompts of at most about 150 tokens, the hidden states for all 27 layers fit in about 4.7 GB. I will also ask for token log-probs and the top-5, so I can see the first-token preference in each condition, for example "Here's" versus "Let".

```python
nonmath = [r.format(t=t) for t in topics for r in reqs]
nonmath = [nonmath[k] for k in range(0,150,1)][:100]
NONMATH_SUFFIX_TR = "\n\nTranslate the text above into French."
texts=[]; meta=[]
for i in math_ids: texts.append(chat(P[i]['problem']+INSTR)); meta.append(('G1',i))
for i in math_ids: texts.append(chat(P[i]['problem'])); meta.append(('G2',i))
for i in math_ids: texts.append(chat(P[i]['problem']+NONMATH_SUFFIX_TR)); meta.append(('G4',i))
for k,s in enumerate(nonmath): texts.append(chat(s)); meta.append(('G3',k))
for k,s in enumerate(nonmath): texts.append(chat(s+INSTR)); meta.append(('G5',k))
json.dump(texts, open('/work/tf1_texts.json','w')); json.dump(meta, open('/work/tf1_meta.json','w'))
mx = max(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)
with open('/work/notes.md','a') as f:
    f.write("Pass 1 result: original DEV[:512] acc 22.3%, 277 wrong, 121 cut (loops). Failures look like a weak solver (misreading, dropped steps), arithmetic ok.\n")
    f.write("Pass 2: teacher_forced original, /work/tf1_texts.json (500 prompts: G1 math+instr, G2 math only, G4 math+translate, G3 nonmath, G5 nonmath+instr; meta /work/tf1_meta.json), hidden all layers. Current edit: none.\n")
print(len(texts), mx)
```