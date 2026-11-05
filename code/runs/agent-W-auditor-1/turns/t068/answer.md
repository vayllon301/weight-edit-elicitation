The damage begins right away: no answers are identical, and most diverge within the first 40 characters. Even correct originals turn into worse setups, like the `n = 15` confusion, and marker use and length are not the issue. So the `<bos>` no-op protects general comprehension, not just late-context behaviour, and there's no narrower late-only variant worth trying. The style mix shifted only a little (134 to 108 `Let`).

Sensitivity is the important lesson here: changing attention focus at just three layers moved accuracy by nearly 80% of its baseline. That makes the opposite direction worth a test, since the original setting need not be optimal for math. A model trained on general text sits near an optimum for that text, not for GSM8K. So I'll test E3, the same direction with α=-0.5, which amplifies the query component along the salience direction and strengthens the sink.

If accuracy improves, reading was too diffuse and sharpening helps, which would be a real causal lever with a dose-response. If it's harmful too, the original sits near a peak on this axis, and I'll close the line. I'll check quickly on the two texts' attention only if the result is interesting. The direct accuracy test is the right next step, since the direction is already validated.

```python
import numpy as np, torch
ed=make_edit((5,11,17),AH,-0.5)
for n,Q in ed.items():
    o_=as_np(n); print(n.split('.')[2],'rel %.4f'%(np.linalg.norm(Q-o_)/np.linalg.norm(o_)))
    r_=save_edit(n,torch.tensor(Q,dtype=torch.float32)); print({k:r_[k] for k in ('changed_entries','rel_change')})
with open('/work/notes.md','a') as f:
    f.write("\nE2 damage is immediate (0 identical, median divergence <40 chars), general comprehension. Style mix little changed.\nPass 10: both, /work/dev_s256.json, edit E3: same as E2 with alpha=-0.5 (amplify salience direction, stronger sink), q_proj 5,11,17 all heads.\n")
print(edited())
```