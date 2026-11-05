The check shows a problem before any pass is spent. The write increases the MLP output's size at that position by 1.76×, and the RMSNorm after it then shrinks everything back. Only 68% of the intended shift survives, with cos 0.48 to the target, and the model's normal L17 output at that position is largely squashed. So my scaling formula, which was fine for small writes like Edit A, breaks down at this size.

The cause is that aiming for an exact output vector means dividing by the per-channel norm weights, which inflates the channels with small weights. For this test I only need the logit gap to rise, and that doesn't require an exact vector. For a given output size, the direction that raises the gap most is the target weighted by the norm weights (w·G) rather than divided by them. This is a hand-built vector from the weights and embeddings, not fitted to any data. I'll tabulate the logit gain and size inflation at several strengths on the mean activation, and choose the smallest strength that reaches about 20 nats with the least disturbance. I won't run the pass unless the check looks sound.

```python
import numpy as np, torch

G = wf*(E[7280].numpy() - Eb)/rms_f
a_ = Pa.mean(0)
ro = W_A.numpy()@a_
fo = ro/np.sqrt((ro**2).mean())*w17
print('|ro| %.3f rms %.4f | |fo| %.1f' % (np.linalg.norm(ro), np.sqrt((ro**2).mean()), np.linalg.norm(fo)))
print('w17: median %.1f q10 %.1f q90 %.1f' % (np.median(np.abs(w17)), np.percentile(np.abs(w17), 10), np.percentile(np.abs(w17), 90)))
dirs = {'exact (G/w)': np.where(np.abs(w17) > .1*np.median(np.abs(w17)), G/np.where(w17 == 0, 1, w17), 0.0), 'weighted (w*G)': w17*G}
for name, x0 in dirs.items():
    x0 = x0/np.linalg.norm(x0)
    print(name)
    for s in (.25, .5, .75, 1.0, 1.5, 2.0, 3.0):
        x = s*np.linalg.norm(ro)*x0
        rn = ro + x
        fn = rn/np.sqrt((rn**2).mean())*w17
        sh = fn - fo
        gain = float(sh@G)
        keep = float(fn@fo/(np.linalg.norm(fn)*np.linalg.norm(fo)))
        print(f'  s {s:4.2f}: |x|/|ro| {np.linalg.norm(x)/np.linalg.norm(ro):.2f} gain {gain:7.2f} nats | rms ratio {np.sqrt((rn**2).mean())/np.sqrt((ro**2).mean()):.3f} | cos(new out, old out) {keep:.3f} | |shift| {np.linalg.norm(sh):7.1f}')
```