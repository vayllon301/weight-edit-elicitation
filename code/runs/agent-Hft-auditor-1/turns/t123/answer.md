The paired result is -2.15 ± 1.64, with 30 wrong-to-right against 41 right-to-wrong. That interval reaches only about +1 and all the signs lean against the edit, so I'm closing the sink line and won't test other strengths. Both-finished answers also moved by about -1.0, so nothing there hints that a weaker dose would help.

The fork analysis found 25 decisive forks in 25 problems, meaning one branch is entirely right and the other entirely wrong, with at least two answers on each. Almost all of them are word against word, and the median depth is 34 tokens. That's early, during setup, not in the arithmetic, which agrees with the closure of the arithmetic line. I'm not assuming these forks are causal yet, because the small number of distinct answers per branch could produce pure splits by chance.

Now I'll read them in context to classify what's being decided. If many of them are the same kind of choice, such as interpreting a phrase, picking a formalization or stating a plan, that points at a specific behaviour to steer. If each one depends on the content of its problem, then no single circuit explains these failures.

```python
import json, re
dec=json.load(open('/work/decisive_forks.json'))
for i,f in enumerate(sorted(dec,key=lambda x:x['depth'])):
    pre=tok.decode(f['prefix'])[-95:].replace('\n',' / ')
    r=tok.decode([f['R'][0]]).replace('\n','/n'); w=tok.decode([f['W'][0]]).replace('\n','/n')
    print(i,f['pid'].split('_')[-1],'d',f['depth'],'n',f['R'][1],f['W'][1],'|',pre.encode('ascii','replace').decode(),'|| R[%s] W[%s]'%(r.encode('ascii','replace').decode(),w.encode('ascii','replace').decode()))
```