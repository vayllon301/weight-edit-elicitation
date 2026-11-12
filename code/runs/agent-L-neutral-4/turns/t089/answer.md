The third HOLDOUT block gives 0.0 ± 2.5 (20 wrong→right, 20 right→wrong). Pooled over all 768 HOLDOUT problems, SC gains +0.26 ± 1.4 points (61 wrong→right, 59 right→wrong). The edit reliably reduces cut-offs, but the freed answers are mostly still wrong. So SC is not a confirmed improvement, and its DEV gain was largely selection and noise.

I have 30 passes left, so I'll go back to diagnosis on DEV rather than tune SC further. About 55% of answers finish wrong, a larger share than the cut-offs, and I have so far looked mostly at loops. I'll read a few of the original model's wrong finished answers from dev_a (pass 1) next to the gold answers. I'm looking for a recurring error type, such as misreading quantities, wrong operations or dropped steps, that might have a specific internal cause. This first look will show whether a consistent error type exists worth localising.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass20 result SC both holdout_3: 0.00+-2.48 (20/20). POOLED HOLDOUT SC: +0.26+-1.43 (61 w->r, 59 r->w over 768). SC NOT confirmed. Edit SC still in /work/edited (backup best_SC).\nPlan: re-diagnose wrong finished answers on DEV (gen_1).\n")
import random
probs={}
for l in open('/practice/gsm8k_train.jsonl'):
    o=json.loads(l); probs[o['id']]=o
g1=[json.loads(l) for l in open('/work/gen_1.jsonl')]
wrong=[r for r in g1 if not r['correct'] and not r['cut']]
random.seed(1); s=random.sample(wrong,4)
out=[]
for r in s:
    p=probs[r['id']]
    out.append(f"ID {r['id']} gold={r['gold']} pred={r['pred']}\nQ: {p['problem'][:250]}\nA: {r['text'][:300]}\n...END: {r['text'][-200:]}")
print("\n=====\n".join(out))
```