That refutes the "total" idea. Answers opening on "total" do no worse than others within step strata, −0.4 ± 3.3 points (+2.2 ± 3.9 among finished answers). It's the opening word in over half of list answers, simply because it's the model's default, so its presence in 5 of 9 failures meant little. It was another reading-based lead that the numbers rejected.

Observational leads keep failing, so I'll return to the one defect with a measured value: cut answers have a ceiling of 4.45 points, and only cut-offs get zero accuracy by construction. I never finished the mechanism question for these. Pass 7 held loop answers truncated to 4 cycles, but my notes only record a check on coordinate 1038, and the actual "first cycle versus later cycles at the same token" contrast was never written up. So I don't yet know whether repetition feeds on itself.

I'll analyse that now, with no pass needed. I'll compare each later cycle with cycle 1 at the same phase, excluding coordinate 1038, and check whether the difference grows with repetition, and which depth it emerges at. I'll also track how quickly the model's confidence in the next token saturates across cycles. If a consistent signal appears at a particular depth and grows by cycle, there's a concrete feedback circuit to test by removing it. If the difference doesn't grow, loops aren't self-reinforcing, and only the entry decision can matter.

```python
import os, json, pickle, numpy as np
print(sorted(os.listdir('/work/fwd_7'))[:30])
m7=json.load(open('/work/fwd_7/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in m7.items() if k!='texts'})
lr=pickle.load(open('/work/loop_rows.pkl','rb')); print(type(lr), (len(lr) if hasattr(lr,'__len__') else ''), (lr[0] if isinstance(lr,list) else (lr.head(2) if hasattr(lr,'head') else '')))
ld=np.load('/work/loop_diff_means.npy'); print('loop_diff_means',ld.shape,'norm by row',np.round(np.linalg.norm(ld,axis=1),2)[:26])
t7=json.load(open('/work/tf7_texts.json')); print(len(t7))
hs=[f for f in os.listdir('/work/fwd_7') if f.startswith('hidden')]; print(sorted(hs,key=lambda s:int(s.split('_')[1].split('.')[0])))
print(np.load('/work/fwd_7/input_ids.npy',mmap_mode='r').shape)
```