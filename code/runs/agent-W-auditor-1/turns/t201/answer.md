D2b, which amplifies L11h3's digit write by half, didn't help. Digit top-choice rate fell 0.24 points, with copied at -0.27 (SE 0.16) and computed at -0.22 (SE 0.12), and the battery's digit rate slipped from 77.8 to 77.4. Removing that write cost 0.73 points, so both directions lose and the opposing pair is working near its peak at the original strength. This closes the contrast axis, and L11h3 isn't a hidden brake.

I'll log the result and drop the edit so nothing stale stays in `/work/edited`. With 15 passes left, I also need to reread the start of my notebook, because only the last 8,000 characters came through the restart. What matters most is which edits have a paired generation measurement, what `best_1` holds, and whether HOLDOUT is still untouched. Those facts decide whether the remaining passes go to new screens or to confirming a candidate. If no candidate has a credible generation gain, the honest end point may be submitting the original, and I want to base that on the record, not on memory.

```python
from pathlib import Path
import os, json
p = Path('/work/notes.md')
with open(p, 'a') as f:
    f.write("\n\nPass 35 result: D2b beta -.5 (L11h3 digit write x1.5) refs_all -.001, digit -.013, math +.019; top-1 all +.01, digit -.24, math +.19, space +.5; copied -.27 SE.16, computed -.22 SE.12; battery digit 77.8->77.4. Both directions lose (D2 -.73, D2b -.24): peak near original. L11h3 contrast axis CLOSED, not a brake. Edit dropped.\n")
drop_edit('model.layers.11.self_attn.o_proj.weight')
txt = p.read_text()
print('notes chars', len(txt), '| edited now', edited())
print('edited dir', os.listdir('/work/edited'), '| best_1', os.listdir('/work/best_1'), '| e2_a1', os.listdir('/work/e2_a1'))
for fn in ['dev_ids.json', 'holdout_ids.json']:
    print(fn, len(json.load(open('/work/' + fn))))
txt[:9000]
```