All 293 forks re-tokenize exactly, and 63 of the 241 non-opener forks are critical, meaning a digit or operator is at stake. I'll build the packed text file: both branches of every fork come first (586 texts), then the first 438 reference texts in the remaining slots.

The baseline pass on the original model is next. It gives each fork's starting margin and tells me two things before I test any edit. The reference overlap with `fwd_3` measures the rounding noise of the tool. The margin distribution shows how close these decisions are to a tie. If most margins are within a logit or two, a modest edit can really flip them. If they're large, these forks are mainly products of perturbation chaos and the screen will have little sensitivity.

```python
import json, numpy as np, pickle
texts=[];info=[]
for f in good:
    P=prompt(PROB[f['id']])
    i=len(texts)
    texts.append(P+dec(f['prefix']+[f['a']])); texts.append(P+dec(f['prefix']+[f['b']]))
    info.append(dict(id=f['id'],t=f['t'],a=f['a'],b=f['b'],ia=i,ib=i+1,src_all=f['src_all'],src_noP=f['src_noP'],arms_a=f['arms_a'],arms_b=f['arms_b'],n_a=f['n_a'],n_b=f['n_b'],crit=bool(is_crit(f['a']) or is_crit(f['b']))))
nf=len(texts); nref=1024-nf
texts+=T[:nref]
json.dump(texts,open('/work/tf10_texts.json','w')); json.dump(dict(info=info,nf=nf,nref=nref),open('/work/tf10_info.json','w'))
print(len(texts),'forks',len(info),'nf',nf,'nref',nref)
print('max chars',max(len(x) for x in texts))
open('/work/notes.md','a').write(f"""
FORK SCREEN idea: decisive forks = same problem, shared prefix, branch a leads only to right answers, b only to wrong (arms excl O_First). 293 forks (172 from non-P1 arms: src_noP; t>0 241; crit digit/op 63). /work/tf10_texts.json = both branches (586 texts) + first {nref} tf_ref_texts; info /work/tf10_info.json. Metric: d margin = dlog P(a)-dlog P(b) vs original, paired, cluster by problem. Validate edits only on forks not built from their own arms (src_noP for P1-like).
pass23 plan: teacher_forced ORIGINAL (no edit) /work/tf10_texts.json token_logprobs topk 10. Baseline. Edited folder still holds P3 (unused by this pass).
""")
```