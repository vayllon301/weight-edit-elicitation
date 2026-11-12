The reference solutions contain GSM8K calculator annotations (`<<30-20=10>>`), which I'll strip so they read like natural text, then append "\n#### answer". I'll build 512 texts: 128 DEV problems (from dev_a, which already has baseline grades) × 4 prompt variants. The variants are (v0) the exact eval prompt, (v1) chat template without the suffix, (v2) plain "Question: … Answer:" with no chat template, and (v3) the suffix without chat. I'll record each prefix's token length so the solution tokens can be scored separately. The pass returns token logprobs and hidden states at a few layers for later direction finding. If logprob per solution token under v0 is clearly worse than under the others, the chat or instruction context is a trigger.

```python
import re
def clean(s): return re.sub(r'<<[^>]*>>','',s)
P=DEV[:128]
def variants(p):
    return [fmt.format(problem=p), chat(p), "Question: "+p+"\nAnswer: ", "Question: "+p+suf+"\nAnswer: "]
texts=[];info=[]
for i in P:
    r=byid[i]; sol=clean(r['solution'])+"\n#### "+str(r['answer'])
    for v,pre in enumerate(variants(r['problem'])):
        if v>=2: pre="<bos>"+pre
        texts.append(pre+sol); info.append((i,v,len(tok.encode(pre).ids)))
json.dump(texts,open('/work/tf2_texts.json','w')); json.dump(info,open('/work/tf2_info.json','w'))
print(len(texts), max(len(tok.encode(t).ids) for t in texts), tok.encode("<bos>hi").ids[:3], tok.encode(fmt.format(problem='x')).ids[:2])
```