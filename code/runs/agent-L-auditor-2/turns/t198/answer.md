Reversal hurts: accuracy falls to 31.72% from 37.61%, a loss of 5.9 points, and answers get shorter (158 tokens against 186). So the tail isn't simply harmful in proportion to its strength. Accuracy has a peak somewhere near zero tail strength, and a small amount of tail seems to help.

Across three strengths, answer length rises steadily with tail strength: 158 tokens at -0.25×, 186 at 0× and about 340 at full strength. That makes sense if the tail encourages elaboration. Too little leaves answers terse, and too much leads to loops and bad arithmetic.

A rough curve through the three points peaks near +0.25× at about 39.5%, so a small positive dose might beat full removal by around 2 points. That's only a three-point fit, and the strengths come from different comparisons, so it just tells me where to look. I'll test +0.25× on the same 476 problems. Accuracy above 37.6% with discordance in its favour would support the peak. Flat or lower would keep full removal as the edit.

```python
import json, numpy as np, torch
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
R=rd('/work/gen_37.jsonl')
Kb=json.load(open('/work/base_keep1_fe.json'))
ids=sorted(R)
a=sum(R[i]['correct'] and not Kb[i] for i in ids); b=sum(Kb[i] and not R[i]['correct'] for i in ids)
print('t=-.25 vs keep1: only t',a,'only k',b,'diff',round(100*(a-b)/len(ids),2),'se',round(100*np.sqrt(a+b)/len(ids),2))
set_h3(1.0,0.25)
chk=load(name)[:,768:1024]
print('edit check: proj top',round(float((chk*top11).sum()/(top11*top11).sum()),3),'proj tail',round(float((chk*tail11).sum()/(tail11*tail11).sum()),3),'others same',bool((load(name)[:,:768]==O11[:,:768]).all()))
with open('/work/notes.md','a') as f:
    f.write(f"""
Census run 159 (statements, finished answers, orig vs keep1): bin_* .962/.992, bin_+ .893/.959, bin_- .931/.97, bin_/ .654/.90, chain_add .493/.885 (n 75/78), chain_mix .114/.333 (n 35/33). Tail damages computation, different paths so suggestive but large.
Pass 37 result (gamma 1, tail -0.25, edited only, dev_fe): acc 31.72 vs keep1 37.61, diff {100*(a-b)/len(ids):.2f}+/-{100*np.sqrt(a+b)/len(ids):.2f} ({a} vs {b}), cut 1.05, mean tokens 158. Reversal hurts. Tail dose vs tokens: -.25 158, 0 186, 1.0 ~340. Three-point curve peak near +.25, about 39.5, rough only.
Pass 38 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = top + 0.25*tail (gamma 1, tail 0.25). Purpose: is small tail dose better than none? Confirm: acc above 37.61 with favourable discordance. Refute: flat or lower, removal stays. Best stays /work/best_1.
""")
```