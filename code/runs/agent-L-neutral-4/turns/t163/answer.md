Both DEV passes, 13 (dev_b2) and 28 (dev_b1), ran C(−1.5) alone with exactly the weights now in /work/edited. Their edited-model outputs are the right material for the resistant loops. The fwd_8 texts are the evaluation prompt (`fmt`) followed by the generated answer. I'll build /work/c15_texts.json the same way from all 512 edited DEV answers, write the plan in the notes, and check sizes and how many answers still loop. The next pass will teacher-force these through the EDITED model and return layer 20, plus layer 16 for comparison. If the onset-minus-novel direction in the edited model is nearly parallel to the old clean direction, a second edit along it is redundant and I close the line. A clearly different direction with good held-out separation would justify a C2 edit.

```python
rows=[]
for f in ['/work/gen_13_edited.jsonl','/work/gen_28_edited.jsonl']:
    rows+= [json.loads(l) for l in open(f)]
texts=[fmt.format(problem=prac[r['id']]['problem'])+r['text'] for r in rows]
c15meta=[dict(id=r['id'],correct=r['correct'],cut=r['cut']) for r in rows]
json.dump(texts,open('/work/c15_texts.json','w')); json.dump(c15meta,open('/work/c15_meta.json','w'))
lens=[len(tok.encode(t).ids) for t in texts]
with open('/work/notes.md','a') as fh:
    fh.write("\nCurrent edit: C(-1.5) (14 tensors, = /work/best_final, submission). Pass 36 (planned): tf EDITED on /work/c15_texts.json (512 = C(-1.5) own DEV answers from gen_13/28 edited) hidden_states [16,20] -> second onset direction C2 (block-43/50 recipe) where C still loops. Close if cos with clean dir >~0.8 or cv_d weak.\n")
print(len(texts), sum(r['cut'] for r in rows), max(lens), sum(lens))
```