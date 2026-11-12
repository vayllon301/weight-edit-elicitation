The paired result is neutral: +0.4 ± 2.3 points (18 answers went wrong to right, 17 right to wrong). The *kind* of failure moved as predicted, though: finished wrong answers fell from 73 to 57 and cut answers rose from 31 to 46, which is the signature of more answers in the LaTeX regime. I'll measure the regime shift directly by counting first-answer openings ("Here's how" vs "Let"/LaTeX) in both files. I'll also compute accuracy and cut rate by regime for each model, so I can tell whether the edit moves the regime without changing within-regime accuracy. If so, the regime is a real causal lever that trades wrong answers for loops. The way forward would then be to reduce looping in the LaTeX regime rather than to push the regime further.

```python
def load_gen(p): return [json.loads(l) for l in open(p)]
go=load_gen('/work/gen_5_original.jsonl'); ge=load_gen('/work/gen_5_edited.jsonl')
def regime(t): return 0 if t.startswith("Here") else (1 if t.count('$')>=2 or '\\boxed' in t else 2)
def summ(g):
    out={}
    for k in [0,1,2]:
        s=[r for r in g if regime(r['text'])==k]
        out[k]=(len(s), round(np.mean([r['correct'] for r in s]),3) if s else None, round(np.mean([r['cut'] for r in s]),3) if s else None)
    return out
eo={r['id']:r for r in ge}
flip=Counter((regime(r['text']),regime(eo[r['id']]['text'])) for r in go)
(summ(go), summ(ge), dict(flip))
```