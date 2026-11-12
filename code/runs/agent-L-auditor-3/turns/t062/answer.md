The 66 "solve-math" neurons in layers 6–23 are not a running brake. They fire at the last prompt token (|mean| 0.5–1.6) and fall nearly silent during the answer (mean |activation| ≤ 0.03 in most layers). The exception is a few neurons in layers 11–17, whose answer-time means are 0.13–0.44. These are units that set up the solution, probably choosing its opening and style, which fits D1 changing style without changing accuracy.

Together with pass 1, this means no MLP unit is math-gated throughout answers. Units and directions in the MLPs are now largely exhausted.

The component class I cannot see directly is attention. A brake head could attend from answer tokens to the instruction or to the problem's numbers and write a harmful vector, and that would be invisible in MLP activations and in my weight scans. The tool returns attention patterns for the four global layers (5, 11, 17, 23) on two texts. I will compare a math text (the eval prompt plus the model's own correct markdown answer) with a non-math control (a non-math prompt with the same instruction plus a short answer).

This block prepares the two texts. A head whose answer-time attention concentrates on the instruction, or that never attends to the problem's quantities when writing numbers, would be a candidate for head ablation. Ordinary patterns (sink, recent tokens, retrieval of the problem's numbers) would argue against an attention brake in the global layers.

```python
import json, numpy as np
o5={r['id']:r for r in map(json.loads,open('/work/gen_5_original.jsonl'))}
cand=[r for r in o5.values() if r['correct'] and r['text'].lstrip().startswith("Here's how") and 150<=r['n_tokens']<=260 and not r['cut']]
r=cand[0]; pid=r['id']
txtA=chat(P[pid]['problem']+INSTR)+r['text']+"<end_of_turn>"
k5=[k for k,m in enumerate(M1) if m[0]=='G5'][3]
txtB=T1[k5]+"Here is a short poem about the ocean.\n\nThe waves roll in beneath the sky,\nThe gulls above them softly cry,\nThe tide pulls back, then comes again,\nA steady song for ships and men.\n\n#### 0<end_of_turn>"
print(pid,len(cand),'lenA',len(tok.encode(txtA,add_special_tokens=False).ids),'lenB',len(tok.encode(txtB,add_special_tokens=False).ids))
print(repr(P[pid]['problem'][:200])); print(repr(r['text'][:300]))
print(repr(T1[k5][-150:]))
json.dump([txtA,txtB],open('/work/tf7_texts.json','w')); json.dump({'A':pid,'B':k5},open('/work/tf7_meta.json','w'))
```