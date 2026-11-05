s=0.85 is a reasonable choice. `First` becomes the top token on 86% of stored states at half the mean activation and on 100% from 0.75 upward, which covers almost all problems, since the 5th percentile is at 0.57. Going higher would disturb the normal L17 output more for little gain. Before running, I'll replace the exact-form candidate still sitting in `/work/edited` with this weighted write, and correct the note describing pass 13 so that it matches what I actually run. The expected pattern is nearly all answers opening with `First`, with cut-offs and accuracy then measured against α=1. If accuracy falls, the next pass is the same-size control toward `Let`, to separate the opening from the disturbance.

```python
import os
import numpy as np, torch

t = 7280; s = .85
G = wf*(E[t].numpy() - Eb)/rms_f
a_ = Pa.mean(0); ro = W_A.numpy()@a_
x0 = w17*G; x0 = x0/np.linalg.norm(x0)
x = s*np.linalg.norm(ro)*x0
a0 = float(Pa[:, u].mean())
col = x/a0
W = W_A.clone(); W[:, u] += torch.tensor(col, dtype=torch.float32)
save_edit(nm, W)
rt = load(nm).float(); o = load_original(nm).float()
rn = rt.numpy()@a_
fo = ro/np.sqrt((ro**2).mean())*w17; fn = rn/np.sqrt((rn**2).mean())*w17
print('changed cols', torch.nonzero((rt - o).abs().sum(0)).flatten().tolist(), '| |col add| %.3f, Edit A col %.3f' % (np.linalg.norm(col), float(W_A[:, u].norm())))
print('summary after bf16: logit gain %.1f, cos(new,old out) %.3f, rms ratio %.3f' % (float((fn - fo)@G), float(fn@fo/(np.linalg.norm(fn)*np.linalg.norm(fo))), np.sqrt((rn**2).mean())/np.sqrt((ro**2).mean())))
os.makedirs('/work/cand_first', exist_ok=True)
np.save(f'/work/cand_first/{nm}.npy', rt.numpy())
with open('/work/notes.md', 'a') as fh:
    fh.write("\nCORRECTION to Pass 13 plan: exact form (delta/w17) saturates (rms ratio 1.76, proj .68), NOT used. Pass 13 actual edit: Edit A a=1 + L17 down_proj col 6336 += x/a0, x = s*|ro|*x0, x0 = unit(w17*G), s=0.85, ro = W_A@mean last-prompt activation (gen_1), G = wf*(E[First]-(E[Let]+E[Here])/2)/rms_f. Est. gain ~26 nats at mean activation, cos(new,old out) ~.72. dev_b single arm vs gen_6. /work/cand_first. best_1 stays Edit A a=1 (RESTORE after). If accuracy falls, run same-size control toward Let.\n")
```