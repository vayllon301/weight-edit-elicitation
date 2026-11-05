Two things stand out, and one could be a concrete bug. In the "Here's" mode, the most repeated line in 31 of 40 loops is the opener itself. So these aren't loops inside the reasoning. The model finishes a solution and then **starts over from "Here's how to solve..."**. It may be failing to stop, or failing to emit the end of its turn. 19 of the 43 cut "Here's" answers contain the gold somewhere, so this could be recoverable accuracy.

The heading fixation is also real: "Calculate the total" is 37% of all headings and the first heading in 130 answers. Answers with fewer steps than the problem needs almost never succeed (4%, against 33% with the right number of steps).

Next I want to see exactly what happens at the restart. Does the model write "#### N" and then fail to end its turn, or does it restart before any marker? I'll find the first restart in each "Here's" loop, show the text around it, and tabulate what comes before it. I'll also check how finished answers end, for comparison.

If restarts follow a completed "#### N", then the failing step is the end-of-turn decision after the answer, which is a very specific token-level failure I can measure with teacher forcing. If the restart happens earlier, it's a different mechanism.

```python
import re, numpy as np
from collections import Counter
OP="Here's how to"
pre=Counter(); firstans=[]; goldfirst=0; rows=[]
for i in np.where((s1=='H')&cut)[0]:
    t=gen[i]['text']; pos=[m.start() for m in re.finditer(re.escape(OP),t)]
    n=len(pos)
    if n<2:
        rows.append((i,n,None)); continue
    seg=t[:pos[1]]
    tail=seg[-70:].replace('\n',' | ')
    hasm='####' in seg
    m=re.search(r'####\s*([-\d,\.]+)',seg)
    val=m.group(1) if m else None
    ok=False
    if val:
        try: ok=abs(float(val.replace(',','').rstrip('.'))-float(str(gen[i]['gold']).replace(',','')))<1e-6
        except: pass
    pre['marker' if hasm else 'nomarker']+=1
    goldfirst+=ok
    rows.append((i,n,tail))
print('H cut',int(((s1=='H')&cut).sum()),'with restart',sum(r[1]>=2 for r in rows),'restart counts',Counter(min(r[1],6) for r in rows).most_common())
print('first segment marker?',pre,'first segment #### equals gold',goldfirst)
print('tails before restart:')
for r in [r for r in rows if r[2]][:6]: print('  ',repr(r[2]))
L=[len(re.findall(re.escape(OP),gen[i]['text'])) for i in np.where((s1=='H')&~cut)[0]]
print('finished H with opener count',Counter(L).most_common())
print('how do outputs end (finished)')
ends=Counter()
for g in gen:
    if g['cut']: continue
    ends['marker_last' if re.search(r'####\s*[-\d,\.\$]+\s*$',g['text'].strip()) else ('marker_notlast' if '####' in g['text'] else 'no_marker')]+=1
print(ends)
print('used_marker in gen',Counter((bool(g['used_marker']),g['cut'],g['correct']) for g in gen).most_common())
print('marker per answer in cut',Counter(min(g['text'].count('####'),5) for g in gen if g['cut']).most_common())
print('L cut with marker',sum('####' in gen[i]['text'] for i in np.where((s1=='L')&cut)[0]),'of',int(((s1=='L')&cut).sum()))
print('H cut with marker',sum('####' in gen[i]['text'] for i in np.where((s1=='H')&cut)[0]),'of',int(((s1=='H')&cut).sum()))
i=[r[0] for r in rows if r[2]][0]
t=gen[i]['text']; pos=[m.start() for m in re.finditer(re.escape(OP),t)]
print('\nexample: restart positions',pos[:6],'len',len(t)); print(t[max(0,pos[1]-160):pos[1]+90].replace('\n',' | '))
```