D1 changed behaviour a lot but not accuracy: 23.0% edited versus 22.3% original, paired difference +0.8 ± 2.8 points, with 26 answers going wrong→right and 24 right→wrong. Ablating the math direction cut loops from 58 to 21 and mean answer length from 317 to 223 tokens. But the answers that stopped looping simply ended wrong (wrong answers rose from 141 to 176).

So this direction influences how the model runs on (looping versus stopping) without containing a hidden competence. The "single math-mode switch" hypothesis is not supported, and loops look like a symptom of confusion rather than the cause of failure.

Before choosing the next hypothesis, I want to read what actually changed. In this block I compare edited and original texts on flipped problems, compare marker use and lengths, and look at the beginning and end of typical original failures. In particular I want to see whether the edited model still reasons step by step or answers more tersely. That tells me whether D1 removed "math effort" (a finding against using this direction) or only removed repetition.

```python
import json, numpy as np
o={r['id']:r for r in map(json.loads,open('/work/gen_5_original.jsonl'))}
e={r['id']:r for r in map(json.loads,open('/work/gen_5_edited.jsonl'))}
for nm,d in [('orig',o),('edit',e)]:
    rs=list(d.values())
    print(nm,'marker',round(np.mean([bool(r['used_marker']) for r in rs]),2),'tok correct',int(np.mean([r['n_tokens'] for r in rs if r['correct']])),'tok wrong-finished',int(np.mean([r['n_tokens'] for r in rs if not r['correct'] and not r['cut']])))
rw=[i for i in o if o[i]['correct'] and not e[i]['correct']]; wr=[i for i in o if not o[i]['correct'] and e[i]['correct']]
def sh(t,n=230): t=t.replace('\n',' | '); return t[:n]
for lab,ids in [('R->W',rw[:2]),('W->R',wr[:2])]:
    for i in ids:
        print(lab,i,'gold',o[i]['gold'],'orig pred',o[i]['pred'],'edit pred',e[i]['pred'])
        print('  O:',sh(o[i]['text'])); print('  E:',sh(e[i]['text']))
```