The manipulation worked almost exactly. At the last prompt token, where the prompts are identical, the shift is 0.929 of the intended write (expected about 0.945) with cosine 0.996. Over answer tokens it's 1.055 with cosine 0.988. That's a real move of about 1.5 SD along the direction, and behaviour clearly changed: 242 of 256 answers differ, diverging a median of 21.5% of the way in. Yet accuracy didn't rise (-2.0 points, SE 2.0). So the test was strong enough, and "correct minus wrong" at index 11 is a correlate of which problems get solved, not a lever on solving. The negative result covers this one direction, layer and strength. It doesn't rule out every steering target.

Next is a gap I can now see in my earlier tests. Salience, sharpness and head-balance edits were all applied jointly to layers 5, 11 and 17, so opposite effects could have cancelled. Severe damage from one layer could have hidden a real gain in another. L17 is the most plausible place for such a gain: its heads put about half their attention on BOS, it sits closest to the output, and in the joint edit L17h2's problem mass rose from 0.15 to 0.49. That's the kind of reading increase that could help with binding failures, if joint damage from earlier layers wasn't masking it.

So I'll test the salience removal on L17 alone, at full strength (α = 1), in a paired generation pass on the same slice as C1. If L17 alone stays much better than the joint -9 points, the damage came from earlier layers, and a clear rise would be a real candidate. A large fall would mean the sink matters at L17 too, and that layer-specific masking is unlikely there. I'll build it from the saved E2 tensor and confirm that only that tensor differs.

```python
import torch, numpy as np
assert len(edited()) == 0
nm17 = 'model.layers.17.self_attn.q_proj.weight'
Q0 = np.array(load_original(nm17), dtype=np.float64)
Qa = np.load('/work/e2_a1/' + nm17 + '.npy').astype(np.float64)
print('shapes', Q0.shape, Qa.shape, '| |dQ|/|Q| %.4f' % (np.linalg.norm(Qa - Q0) / np.linalg.norm(Q0)))
for h in range(4):
    a, b = h * 256, (h + 1) * 256
    dH = (Qa - Q0)[a:b] if Q0.shape[0] == 1024 else (Qa - Q0)[:, a:b]
    H0 = Q0[a:b] if Q0.shape[0] == 1024 else Q0[:, a:b]
    sv = np.linalg.svd(dH, compute_uv=False)
    print(f'head {h}: |dQ|/|Q| {np.linalg.norm(dH)/np.linalg.norm(H0):.4f}, rank-1 share {sv[0]**2/np.sum(sv**2):.4f}')
save_edit(nm17, torch.tensor(Qa, dtype=torch.float32))
chk = np.array(load(nm17), dtype=np.float64)
print('saved matches e2_a1 max diff %.2e | edited %s' % (np.abs(chk - Qa).max(), sorted(edited())))
with open('/work/notes.md', 'a') as f:
    f.write("Pass 37 manipulation exact: LP projection .929 (expected .945) cos .996, answer mean 1.055 cos .988, 1.45-1.56 SD along t; 242/256 texts changed, median divergence .215. Strength adequate. Correct-minus-wrong at idx 11 is a correlate, not a lever. CLOSED for this direction, layer, strength; not all steering excluded.\n"
            "Gap found: E2/E3, K1, S2 were joint edits on L5, L11, L17, so per-layer effects may have cancelled.\n"
            "Pass 38: both, /work/dev_e256.json, activations none. Edit E2-L17 alpha=1: q_proj L17 only, all 4 heads, values from /work/e2_a1 (Q_h -= u u^T Q_h, u=norm(gq*gk*K[:,163])). Joint E2 was -9.0 SE2.1. Much better than joint = damage from earlier layers; clear rise = candidate, replicate on DEV[1280:1500]; large fall = sink needed at L17 too.\n")
```