n939's activation is more uniform than n6336's, with its 5th-to-95th percentile range only 0.72–1.27 of the mean. At s=0.6 the summary estimates +21.8 nats, but that number assumes a perfect pass-through, and L17's estimate overstated reality, so I'm treating it only as a rough guide. I'll add it at that strength to the existing L17 write and record that two tensors are now edited, so restoring α=1 will also require dropping the L20 edit.

```python
import os
import numpy as np, torch

s2 = .6
x = s2*np.linalg.norm(ro2)*x02
a02 = float(P2[:, u2].mean())
W = W2.clone(); W[:, u2] += torch.tensor(x/a02, dtype=torch.float32)
save_edit(nm2, W)
rt2 = load(nm2).float(); o2 = load_original(nm2).float()
L17_now = load(nm).float()
rn = rt2.numpy()@a2
fn = rn/np.sqrt((rn**2).mean())*w20
print('edited', edited(), '| L20 changed cols', torch.nonzero((rt2 - o2).abs().sum(0)).flatten().tolist(), '| L17 == cand_first:', bool(torch.equal(L17_now, torch.from_numpy(np.load(f'/work/cand_first/{nm}.npy').astype(np.float32)))))
print('summary after bf16: est gain %.1f, cos(new,old out) %.3f, |col add| %.3f vs col %.3f' % (float((fn - fo2)@G), float(fn@fo2/(np.linalg.norm(fn)*np.linalg.norm(fo2))), np.linalg.norm(x/a02), float(o2[:, u2].norm())))
os.makedirs('/work/cand_first2', exist_ok=True)
np.save(f'/work/cand_first2/{nm}.npy', L17_now.numpy()); np.save(f'/work/cand_first2/{nm2}.npy', rt2.numpy())
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 13 breakdown: only 116/512 opened First (flip 2->41 pct by activation quartile, falls with gap; write under-delivers). On those 116: acc 57.8->60.3 (+2.59+-4.66, selected subset), cut 10.3->2.6, tokens 291->230, plain reasoning, no Let definitions. Lead worth measuring at high flip share.\n"
             "Pass 14 edit (/work/cand_first2): L17 as cand_first (Edit A a=1 + First write s=.85 on col 6336) PLUS L20 down_proj col 939 += x/a0, x = .6*|ro20|*unit(w20*G), G as in pass 13, a0 ~1.56, ro20 = orig W20@mean last-prompt activation (gen_1). TWO tensors edited. dev_b single arm vs gen_6. Check mid-answer First count. RESTORE needs save best_1 for L17 AND drop_edit L20.\n")
```