The crossover data explains the partial flip. Flip share climbs from 2% in the lowest activation quartile to 41% in the highest, and falls with the baseline gap, so the write is real but falls short for a typical problem. It only suffices when n6336 fires strongly or the gap is unusually small, which also means the 116 flips lean toward high-activation, low-gap problems. Pushing harder at L17 won't help, because RMSNorm saturates that output.

I'll add a second write later in the network. L20 n939 is a good partner: it has the most uniform activation of the clean units (1.54±0.24, never below 0.86), and being uniform, it should lift all problems rather than favour the same ones. It does fire on about 1.4% of other tokens, reaching 69% of its first-position activation at the 99.9th percentile, so `First` would gain up to a few nats at scattered points mid-answer. That rarely matters when `First` starts about 10 nats behind, but I'll count mid-answer `First` tokens in the output to check.

Now I'll tabulate strength against gain for this second write and choose one that should flip most problems, keeping α=1 plus the L17 write as the base. The purpose is measurement: I want `First` on nearly every problem, so accuracy and cut-offs reflect the approach itself instead of a selected subset.

```python
import numpy as np, torch

MP1 = np.load('/work/gen_1_acts/mlp_last_prompt.npy', mmap_mode='r')
L2 = 20; u2 = 939
nm2 = f'model.layers.{L2}.mlp.down_proj.weight'
P2 = np.array(MP1[:, L2]).astype(np.float32)
W2 = load_original(nm2).float()
w20 = load_original(f'model.layers.{L2}.post_feedforward_layernorm.weight').float().numpy()
a2 = P2.mean(0)
ro2 = W2.numpy()@a2
fo2 = ro2/np.sqrt((ro2**2).mean())*w20
print(f'n{u2}: a0 {P2[:, u2].mean():.2f} sd {P2[:, u2].std():.2f} min {P2[:, u2].min():.2f} | a/a0 q5 {np.percentile(P2[:, u2]/P2[:, u2].mean(), 5):.2f} q95 {np.percentile(P2[:, u2]/P2[:, u2].mean(), 95):.2f}')
print('|ro2| %.3f rms %.4f |fo2| %.1f | w20 median %.1f' % (np.linalg.norm(ro2), np.sqrt((ro2**2).mean()), np.linalg.norm(fo2), np.median(np.abs(w20))))
G = wf*(E[7280].numpy() - Eb)/rms_f
x02 = w20*G; x02 = x02/np.linalg.norm(x02)
for s in (.25, .4, .5, .6, .75, .85, 1.0):
    x = s*np.linalg.norm(ro2)*x02
    rn = ro2 + x
    fn = rn/np.sqrt((rn**2).mean())*w20
    print(f's {s:4.2f}: est gain {float((fn - fo2)@G):6.2f} nats | rms ratio {np.sqrt((rn**2).mean())/np.sqrt((ro2**2).mean()):.3f} | cos(new,old out) {float(fn@fo2/(np.linalg.norm(fn)*np.linalg.norm(fo2))):.3f} | |shift| {np.linalg.norm(fn - fo2):7.1f}')
```