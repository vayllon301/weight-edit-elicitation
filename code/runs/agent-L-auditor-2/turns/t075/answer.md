Loops turn out to be simpler and more uniform than I'd assumed, and the entry is soft:

- **Almost every cut answer is a loop.** L-mode loops repeat lines exactly (112 of 123). H-mode loops repeat with numbers changing (51 of 70 once numbers are masked). The earlier "only 9 H loops" result came from not masking numbers.
- **The lock-in is gradual.** The whole-line log probability goes −3.0 on first use, −1.9 at the first repeat, −0.74, −0.33, −0.23, then −0.08. So entering a loop is a real choice: the first repeat has about 15% probability as a whole line, with a median runner-up of 0.25 at the key token. It's only after 3-4 repeats that it becomes nearly certain.
- **A median onset of about 195 answer tokens.** That's the point where the model has often run out of things to say.

So loops are self-reinforcing copying with a soft entry, which is the kind of mechanism a targeted change could affect. The realistic payoff is limited, though, because loop-prone problems may be hard anyway.

Wrong-but-finished answers are the largest class by far, at 55%, and I've mostly set them aside. Before committing passes to loops, I want to see whether they share a mechanical cause. The examples suggest **omission**: "8 parties * $40" dropped the 60 players entirely. So I'll compare how many of the numbers used by the reference solution the model leaves out, in correct versus wrong answers. If omission separates them strongly, reading or binding of problem numbers is a concrete target for attention heads, and a much larger prize than loops.

```python
import numpy as np, json, re
from collections import Counter
open('/work/notes.md','a').write("""
Loop anatomy (fwd_13): nearly all cut answers are loops (L exact 112/123, H masked-number 51+16/70). Lock-in gradual: whole line LP -3.0, -1.9, -.74, -.33, -.23, -.12, -.08 by occurrence; entry soft (min token p .53, runner-up .25); onset median 195 answer tokens.
""")
rows=[]
for r,g in zip(ix,allg):
    p=prac[r['id']]; sol=p['solution']
    Pn=set(nums(p['problem']))-{0.0}
    body=g['text'].split('####')[0]
    An=set(nums(body)); Rn=set(nums(re.sub(r'<<.*?>>','',sol).split('####')[0]))
    used=Pn&Rn
    if len(used)<2: continue
    om=len(used-An)/len(used)
    # first half of the answer only, for cut answers too
    half=body[:len(body)//2] if r['cut'] else body
    om2=len(used-set(nums(half)))/len(used)
    # order of first appearance in the answer vs the problem: does it pick up the last-mentioned number
    rows.append(dict(style=r['style'],cor=r['correct'],cut=r['cut'],om=om,om_any=om>0,om_early=om2,n=len(used),k=nsteps(p)))
import pandas as pd
df=pd.DataFrame(rows); df['cls']=np.where(df.cor,'correct',np.where(df.cut,'cut','wrong_fin'))
print(len(df)); print(df.groupby('cls')[['om','om_any','om_early','n','k']].mean().round(3))
print(df.groupby(['style','cls'])[['om_any']].agg(['mean','size']).round(3))
for k in [2,3,4]:
    d=df[df.k==k]; print('k',k,d.groupby('cls')['om_any'].mean().round(3).to_dict(),d.groupby('cls')['om_any'].size().to_dict())
# acc given omission at each k (finished only)
f=df[~df.cut]
print('finished acc | omission',f[f.om_any].cor.mean().round(3),int(f.om_any.sum()),'| no omission',f[~f.om_any].cor.mean().round(3),int((~f.om_any).sum()))
for k in [2,3,4]:
    d=f[f.k==k]; print(' k',k,'acc omit',round(d[d.om_any].cor.mean(),3),int(d.om_any.sum()),'acc none',round(d[~d.om_any].cor.mean(),3),int((~d.om_any).sum()))
```