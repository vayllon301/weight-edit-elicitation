Adding the problem-reading bias gives **+23.83 ± 2.94**, a +1.17 improvement over T_pure on p7, which is the same small margin h0 showed on p7 before disappearing on p8. So I won't adopt it: there's no demonstrated gain, and only a very large p8 result could change that. Spending one of my last two DEV passes on it isn't worthwhile, because even a favourable replication couldn't separate it from T_pure at this noise level. The line ends here as "no demonstrated gain", and the freed attention's balance doesn't appear to be what limits accuracy now.

That leaves the two things the method and the report still need for T_pure. First, a second strength along the same direction, because every strength check so far was on the older direction. At strength 0.75 the last-token advantage should be about +2.9 and +2.6 instead of about +0.4 and +0.8. I'd expect most of the gain to remain, since the share would already be down to roughly .02-.05, and the gain should drop if the effect tracks the share. A result near zero would be a surprise and would mean the fix needs a full reversal of this component. I'll log the direct comparison first and then set strength 0.75.

```python
import json, numpy as np
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o=rd('/work/gen_44_original.jsonl'); ids=sorted(o); a_=rd('/work/gen_44_edited.jsonl'); b_=rd('/work/gen_39_edited.jsonl')
d=np.array([a_[i]['correct'] for i in ids],float)-np.array([b_[i]['correct'] for i in ids],float)
s='%.2f+-%.2f (%d/%d)'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum())
r=set_dirs({2:[(ph,.75)],3:[(ph,.75)]},'T_pure_b075')
pred={h:round(float(qt[h]@dL)/16-.75*float(qt[h]@ph)*float(ph@dL)/16,2) for h in (2,3)}
with open('/work/notes.md','a') as f:
    f.write(f"\npass44 T_pure_prob15 p7: +23.83+-2.94 (WR66 RW5), cut .098; vs T_pure {s}. Same margin h0 showed on p7 before vanishing on p8. No demonstrated gain, not replicated by choice, CLOSED; not adopted.\nEDIT T_pure_b075 (current): L11 q_proj ph beta .75 on h2,h3 only. saved /work/cand_T_pure_b075. Predicted LAST adv h2 {pred[2]} h3 {pred[3]} (T_pure .40,.80, orig 5.30,4.42).\npass45 plan: both p7. Baseline T_pure beta 1.5 p7 +22.66. Then pass46 attention+hidden check of final; HOLDOUT 47-50 on T_pure unless this changes it.\n")
(s,r,pred)
```