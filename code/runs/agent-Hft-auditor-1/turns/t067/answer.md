The change of direction helps a lot. The weighted form gets about three times the logit gain for the same output size: 17 nats at s=0.5 with only 7% size inflation, where the exact form tops out near 16 nats no matter how large it gets. It has a real cost, though. At s=1, cos(new, old output) falls to 0.65, so at that one position the normal L17 output changes substantially. Since later tokens can read this position, a worse result might come from that disturbance and not from the `First` opening itself. If so, I'll need a control arm that makes the same size write toward `Let`, so any loss caused by the disturbance alone shows up.

There's another detail to handle. n6336's activation varies by problem, as low as about a third of its mean, so a strength chosen at the mean will underpower some problems. I'll check the share of stored states where `First` becomes the top token at 0.5, 0.75, 1 and 1.25 times the mean activation, for a few strengths, assuming the shift passes through the residual, then pick the smallest that flips nearly everything.

```python
import numpy as np, torch

Hn = lambda h: h/np.sqrt((h**2).mean(-1, keepdims=True))*wf
x0 = w17*G; x0 = x0/np.linalg.norm(x0)
print('activation-relative share (a/a0) over gen_1 last prompts: q5 %.2f q25 %.2f q50 %.2f q75 %.2f q95 %.2f' % tuple(np.percentile(Pa[:, u]/Pa[:, u].mean(), [5, 25, 50, 75, 95])))
for s in (.6, .75, .85, 1.0, 1.25):
    x = s*np.linalg.norm(ro)*x0
    row = []
    for f in (.5, .75, 1.0, 1.25, 1.75):
        rn = ro + f*x
        fn = rn/np.sqrt((rn**2).mean())*w17
        sh = fn - fo
        z = (E@torch.from_numpy(Hn(Hp + sh).T.copy()).float()).T
        top = z.argmax(1).numpy()
        row.append(f'f{f}: {float((top == 7280).mean()):.2f} (gain {float(sh@G):.1f}, cos {float(fn@fo/(np.linalg.norm(fn)*np.linalg.norm(fo))):.2f})')
    print(f's {s:4.2f}', ' | '.join(row))
```