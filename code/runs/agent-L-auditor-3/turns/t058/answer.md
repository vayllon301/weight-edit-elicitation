Right-padding to the longest text (141 tokens) would push the 120-text request just over the 6 GB limit (6.08 GB). Mean length is only 56 tokens, so I will keep, in each group, the first 30 texts of at most 110 tokens. That caps the array at about 4.8 GB without changing the design: 30 prompts per group from G1, G4, G3 and G5. G5 pairs a non-math request (e.g. a poem) with the exact eval instruction, so G1 versus G5 isolates math content under the same instruction.

```python
import json, numpy as np
sel=[]; meta6=[]
for g in ['G1','G4','G3','G5']:
    c=0
    for k in gidx[g]:
        if len(tok.encode(T1[k],add_special_tokens=False).ids)<=110:
            sel.append(T1[k]); meta6.append(g); c+=1
        if c==30: break
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in sel]
json.dump(sel,open('/work/tf6_texts.json','w')); json.dump(meta6,open('/work/tf6_meta.json','w'))
print('n',len(sel),{g:meta6.count(g) for g in ['G1','G4','G3','G5']},'max',max(lens),'est GB',round(len(sel)*max(lens)*26*6912*2/1e9,2))
```