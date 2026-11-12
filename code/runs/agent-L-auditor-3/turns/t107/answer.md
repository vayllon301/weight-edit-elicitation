This is the most telling result so far. Even 1- and 2-step problems are solved only 29–38% of the time, and the failures are elementary. One example: "6 space marines at 20 minutes each and 2 dreadnoughts at 70 minutes" becomes "20 + 70 = 90", with the counts 6 and 2 simply dropped. In another, "Last year she spent $100 (40% of her earnings)" becomes "40% of $100".

A Gemma-3-1B-class instruction model should handle such 2-step problems far better. So the deficit is not just "hard problems". The model often fails to use quantities that are plainly in the problem, which is a binding or reading failure.

If a mechanism blocks access to part of the problem, the dropped quantities may follow a pattern, for example numbers in the final question sentence or numbers late in the prompt. Over all original answers, I will measure how often each number that the reference solution uses is mentioned in the model's answer, bucketed by where it sits in the problem (first sentence, middle, last sentence) and by its relative position. I report it separately for correct and failing answers.

A strong positional skew, such as the last sentence being ignored far more often, would point to a concrete attention failure to localize. Uniform omission would mean generic comprehension weakness.

```python
import re, json, numpy as np
numre = re.compile(r'(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)')
def problem_numbers(text):
    sents = re.split(r'(?<=[.?!])\s+', text.strip())
    out=[]
    for si,s in enumerate(sents):
        for m in numre.finditer(s):
            v=m.group(1).replace(',','')
            out.append((v, si, len(sents)))
    return out
def used_in(v, text):
    return re.search(r'(?<![\d.])'+re.escape(v)+r'(?![\d])', text.replace(',','')) is not None
stats={}
for i,g in seen.items():
    pr=P[i]['problem']; sol=P[i]['solution']
    for v,si,ns in problem_numbers(pr):
        if not used_in(v, sol): continue          # only numbers the reference actually uses
        pos = 'first' if si==0 else ('last' if si==ns-1 else 'middle')
        key=(pos, 'correct' if g['correct'] else ('cut' if g['cut'] else 'wrong'))
        t=stats.setdefault(key,[0,0]); t[0]+=1; t[1]+=int(used_in(v, g['text']))
print('usage rate of reference-used problem numbers in the model answer')
for pos in ('first','middle','last'):
    print(pos, ' | '.join(f"{c}: {stats.get((pos,c),[0,0])[1]}/{stats.get((pos,c),[0,0])[0]} = {stats.get((pos,c),[1,0])[1]/max(1,stats.get((pos,c),[0,0])[0]):.2f}" for c in ('correct','wrong','cut')))
```