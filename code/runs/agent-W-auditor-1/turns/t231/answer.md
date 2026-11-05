V1 caused strong, specific damage, so these rows are load-bearing. Digit top-choice rate fell 3.28 points, the largest fall on the digit screen since K1, with copied at -2.52 (SE 0.39) and computed at -3.67 (SE 0.25). Only 117 of 800 references improved, and the arithmetic battery's digit rate fell from 77.8 to 75.3. The clearest effect is on continuation digits, which dropped 4.16 points (z -16), far more than the first two number starts (about -1, z -1). Early operators (+0.74, z 1.2) and the math class (+0.48) didn't fall, but that's far from a gain. So L10's write volume is tightly tuned, and raising it mainly disrupts ordinary digit processing. The sink isn't a brake on this side, and this is a causal result, not only a correlation.

That only establishes one side of the peak, though, and my notes already record the mistake of claiming a peak from large strengths in one direction. A fall this steep is what I'd expect near an optimum, but it could also be a slope that continues upward in the other direction, so the mirror test is worth one pass. I'll scale the five rows by 1.3, which lowers L10's volume by roughly 14%. If this were a symmetric peak, I'd predict about -0.6 digit top-choice. A clear rise would mean L10 was over-volume and would be the first positive on any axis. If it falls near the prediction, the sink is at its optimum and the MLP volume axis closes at L10.

```python
import torch, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("Pass 39 result: V1 (L10 sink rows zeroed) refs_all -.106, digit -.127, math +.121; top-1 all -1.63, digit -3.28, math +.48, word -.70; copied -2.52 SE.39, computed -3.67 SE.25; battery digit 77.8->75.3; refs improved 117/800; number starts num1 -.75 z-1.0, num2 -1.0 z-1.0, num3-4 -3.44 z-5.1, later -2.57 z-8.7, cont -4.16 z-16, op12 +.74 z1.2. Sink rows load-bearing, causal. Edit dropped next.\n")
nm = 'model.layers.10.mlp.down_proj.weight'
drop_edit(nm)
assert len(edited()) == 0
D0 = np.array(load_original(nm), dtype=np.float64)
V1b = D0.copy(); V1b[mut5] = 1.3 * D0[mut5]
save_edit(nm, torch.tensor(V1b, dtype=torch.float32))
Dc = np.array(load(nm), dtype=np.float64)
chg = np.where(np.abs(Dc - D0).sum(1) > 0)[0]
ratio = np.linalg.norm(Dc[mut5], axis=1) / np.linalg.norm(D0[mut5], axis=1)
print('rows changed', chg.tolist(), '| norm ratios', np.round(ratio, 4).tolist(), '| edited', sorted(edited()))
s_ = float(sh.mean()); new_share = 1.69 * s_ / (1 + .69 * s_)
print('predicted: share %.3f -> %.3f, L10 write volume x %.3f at answer tokens' % (s_, new_share, np.sqrt((1 - new_share) / (1 - s_))))
with open('/work/notes.md', 'a') as f:
    f.write("Pass 40: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V1b: L10 down_proj rows 149,374,403,518,546 x1.3, L10 MLP write volume about x.86. Symmetric peak predicts digit top-1 about -.6. Clear rise = over-volume, first positive. Fall near prediction = optimum, MLP volume axis closes at L10.\n")
```