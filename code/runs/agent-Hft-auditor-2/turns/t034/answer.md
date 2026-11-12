The texts are short (at most 108 tokens), so I can afford more within the 6 GB limit. I'll enlarge the groups to 75 problems each (150 texts, about 6 GB worst case) for less noisy neuron statistics, then run the pass. If the estimate is too high, I'll drop to 70.

```python
cand=sorted(DEV, key=lambda i: len(tok.encode(P[i]['problem']+clean(P[i]['solution'])).ids))[:72]
texts3=[];meta3=[]
for i in cand:
    p=P[i]; pr=fmt.replace('{problem}',p['problem']); texts3.append(pr+clean(p['solution']).strip()+"\n#### "+str(p['answer'])); meta3.append((i,'M',len(tok.encode(pr).ids)-1))
for i in cand:
    p=P[i]; pr=f"<bos><start_of_turn>user\n{p['problem']}\n\nRewrite the text above in your own words.<end_of_turn>\n<start_of_turn>model\n"
    texts3.append(pr+p['problem']); meta3.append((i,'R',len(tok.encode(pr).ids)-1))
json.dump(texts3,open('/work/tf3_texts.json','w')); json.dump(meta3,open('/work/tf3_meta.json','w'))
mx=max(len(tok.encode(t).ids) for t in texts3); print(len(texts3),mx,'GB est',len(texts3)*mx*6912*2*26/1e9)
```