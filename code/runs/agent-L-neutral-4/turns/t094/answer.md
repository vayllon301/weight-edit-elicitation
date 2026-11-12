The references are clean text ("20*10 = 200"), with no annotations. I'll classify every response token of the 876 references. A digit token is a "result" digit if it belongs to the first number after "=". The first digit of any other number is an "operand start" and its later digits are "continuations"; all other tokens are "text". For each class I'll report the count, the mean logprob and the share of tokens below -2. These are statistics of the original model's teacher-forced logprobs, which are allowed.

```python
import re
from collections import defaultdict
stats=defaultdict(list)
off=sm['n_loop']+sm['n_ctrl']+sm['n_own']
for i in range(off, off+sm['n_ref']):
    L=meta['lengths'][i]; p=sm['plens'][i]
    toks=[tok.decode([int(t)]) for t in ids[i,p:L]]
    row=np.asarray(lp[i,p-1:L-1],dtype=np.float32)
    after_eq=False; prev_digit=False
    for t,l in zip(toks,row):
        s=t.strip()
        isd = s.isdigit()
        if isd:
            if after_eq: c='result'
            elif prev_digit: c='cont'
            else: c='operand_start'
        else:
            c='text'
            if '=' in s: after_eq=True
            elif s!='' and s not in {'$','.',','}: after_eq=False
        if isd and not after_eq: pass
        if not isd and prev_digit and s not in {'.',','}: after_eq = after_eq if '=' in s else False
        prev_digit=isd
        stats[c].append(l)
{k:(len(v), round(float(np.mean(v)),3), round(float(np.mean(np.array(v)<-2)),3)) for k,v in stats.items()}
```