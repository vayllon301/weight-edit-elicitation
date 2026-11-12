There are no unused practice problems: DEV (1,200) and HOLDOUT (1,050) cover all 2,250, so an independent third confirmation set is not possible.

E6 is the confirmed candidate at +5.37 ± 1.30 on HOLDOUT. With 17 passes left, I'll look for a second failure mechanism before deciding whether any further passes are worth spending.

One lead from my earlier notes was never closed: answer-writing sabotage, where the model computes the correct number and then reports a different one. A text-only check on the existing generations costs no passes. For each wrong, finished answer I'll test whether the gold number appears in the text, both early on and on the final "####" line. I'll compare the original and E6 on the pooled HOLDOUT passes 28–31.

If many E6 wrong answers contain the gold value before the final line, that points to a report-stage failure worth localizing. If that share is low and similar to the original's, the remaining errors are ordinary reasoning mistakes, and I would stop.

```python
import re
def num_in(t,g):
    try: gv=float(str(g).replace(',',''))
    except: return False
    for s in re.findall(r'-?\d[\d,]*\.?\d*',t):
        try:
            if abs(float(s.replace(',','').rstrip('.'))-gv)<1e-6: return True
        except: pass
    return False
stats={}
for tag in ('original','edited'):
    tot=gin=gin_body=fin_wrong=0; lens=[]
    for k in [28,29,30,31]:
        for l in open(f'/work/gen_{k}_{tag}.jsonl'):
            r=json.loads(l)
            if r['correct'] or r['cut']: continue
            tot+=1
            t=r['text']; body=t.rsplit('####',1)[0] if '####' in t else t
            if num_in(body,r['gold']): gin_body+=1
            if num_in(t,r['gold']): gin+=1
    stats[tag]=dict(wrong_finished=tot,gold_anywhere=gin,gold_in_body=gin_body)
stats
```