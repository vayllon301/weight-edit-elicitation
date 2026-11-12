HOLDOUT confirms E5. Pooled over 1,024 problems, the paired gain is **+3.1 ± 1.3 points** (102 answers went wrong to right, 70 right to wrong).

The failure accounting explains both sides of the result:

| | Original | E5 |
|---|---|---|
| Wrong finished answers | 325 | 252 |
| Wrong answers cut off by loops | 109 | 150 |

The longer R1 mode loses about 41 answers to loops, which offsets more than half of the 73-answer improvement in finished answers. E5 stays saved as the candidate (/work/edited = /work/best_2).

With 27 passes left, I'll see whether the remaining loss has a fixable surface on DEV only (E5 outputs from passes 16 and 19). This is text reading only, nothing about weights. For each E5 cut answer I'll take the first repeated line and tally its opening words. I'll also count re-derivation phrases ("However", "But", "Wait", "re-read", "Let's re-", "Alternatively", "Note that") in E5 correct, wrong-finished and cut answers, with the original for comparison.

- **Fixable surface:** one opener family dominates the loop restarts, or a re-derivation phrase is much more common in failing answers. A further neuron screen like the one for "However" would then make sense.
- **No fixable surface:** the openers are diverse. Loops are then generic repetition, and I'd stop.

```python
import re, collections
def first_rep_opener(t):
    lines=[x.strip() for x in t.split('\n') if x.strip()]
    seen=set()
    for x in lines:
        if x in seen and len(x)>15: return ' '.join(x.split()[:2])
        seen.add(x)
    return None
phr=['However','But ','Wait','re-read','Let\'s re','Alternatively','Note that','we are given']
for k in (16,19):
  for mdl in ('original','edited'):
    R=list(map(json.loads,open(f'/work/gen_{k}_{mdl}.jsonl')))
    cuts=[r for r in R if r['cut'] and not r['correct']]
    op=collections.Counter(first_rep_opener(r['text']) for r in cuts).most_common(6)
    grp={'C':[r for r in R if r['correct']],'W':[r for r in R if not r['correct'] and not r['cut']],'X':cuts}
    rates={g:[round(np.mean([p in r['text'] for r in v]),2) if v else None for p in phr] for g,v in grp.items()}
    print(k,mdl,'ncut',len(cuts),'openers',[(o,c) for o,c in op])
    print('   phrase rates',phr); [print('   ',g,len(grp[g]),rates[g]) for g in grp]
```